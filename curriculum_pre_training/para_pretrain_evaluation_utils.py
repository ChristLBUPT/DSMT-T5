try:
    from .para_pretrain_multitask_dataset import ParaPretrainMultitaskDataset, ParaPretrainMultitaskValDataset, TaskFormatter
except ImportError:
    import sys; sys.path.append('..')
    from para_pretrain_multitask_dataset import ParaPretrainMultitaskDataset, ParaPretrainMultitaskValDataset, TaskFormatter
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import re
import tqdm
from collections import defaultdict
from multiprocessing import cpu_count, Pool
import subprocess
import h5py
import os
import numpy as np
import json
import nltk
import logging
from typing import Union, List, Literal, Dict
from nltk.translate.bleu_score import corpus_bleu
from evaluation.myevalb import sentence_score as myevalb_sentence_score
from data.TreeUtils.linearizing_tree import bracket_to_slash, slash_to_bracket
from copy import deepcopy
import warnings
from tensorboardX import SummaryWriter
from pandas import DataFrame


def form_sentence(list_of_tokens: List[str]):
    """form a sentence, replacing white chars(0x2581) with space"""
    return ''.join([each.replace('\u2581', ' ') for each in list_of_tokens])

def split_list(l: List[str], delimiter: str):
    """split a list by delimiter"""
    indices = [i for i, x in enumerate(l) if x == delimiter]
    return [l[i + 1:j] for i, j in zip([-1]+indices, indices+[None])]

def calculate_prf(pred: List[List[str]], gt: List[List[str]]):
    """calculate precision, recall and f1 score of a prediction given ground truth, supports redundant elements in prediction and ground truth,
    one element in prediction only matches one element in ground truth, and vice versa"""
    # 1. calculate precision
    flag = [False] * len(gt) # flag means whether a gt is already matched with a pred
    num_correct = 0
    num_pred_spans = len(pred)
    for i, pred_span in enumerate(pred):
        for j, gt_span in enumerate(gt):
            if pred_span == gt_span and not flag[j]:
                flag[j] = True
                num_correct += 1
                break
    
    p = num_correct / num_pred_spans if num_pred_spans > 0 else 0
    # 2. calculate recall
    flag = [False] * len(pred)
    num_correct = 0
    num_gt_spans = len(gt)
    for i, gt_span in enumerate(gt):
        for j, pred_span in enumerate(pred):
            if pred_span == gt_span and not flag[j]:
                flag[j] = True
                num_correct += 1
                break
    
    r = num_correct / num_gt_spans if num_gt_spans > 0 else 0
    # 3. calculate f1 according to p and r
    f1 = (2 * p * r / (p + r)) if p != 0 and r != 0 else 0

    return p, r, f1


class ParaPretrainMultitaskEvaluator:
    def __init__(self, evalb_executable_path: str, evalb_param_file_path: str, tokenizer: T5Tokenizer, tree_format: Literal['bracket', 'slash'] = 'bracket') -> None:
        # self.tasks = TaskFormatter.tasks
        nltk_data_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/liuhongxu02/nltk_data/'
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.append(nltk_data_path)
        self.evalb_executable_path = evalb_executable_path
        self.evalb_param_file_path = evalb_param_file_path
        # self.val_data_dir = val_data_dir
        self.tree_format = tree_format
        # self.supported_tasks = ['constituency_discrimination', 'constituency_searching', 'production_detection', ]
        self.supported_tasks = [
            'constituency_discrimination', 'constituency_searching', 
            'production_detection', 'tree_pruning', 'pruned_tree_parse', 'pruned_tree_completion',
            'treeposition_indexing', 'node_deleting', 'tree_forming', 'height_selection'
        ]

    def evaluate_evalb(self, input_file_path: str, gt_file_path: str):
        outputs = subprocess.Popen([self.evalb_executable_path, '-p', self.evalb_param_file_path, gt_file_path, input_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,).communicate()
        print('stdout:', outputs[0].decode())
        print('stderr:', outputs[1].decode())

    def evaluate_myevalb(self, inputs_trees: List[List[str]], reference_trees: List[List[str]]):
        def join_subtokens_for_tree(subtokens: List[str]):
            res = ' '.join(subtokens).replace(chr(0x2581), ' ').replace('</s>', '')
            if self.tree_format == 'slash':
                res = re.sub(r'/[ ]+', '/', res)
            return res
        inputs_trees = [*map(join_subtokens_for_tree, inputs_trees)]
        reference_trees = [*map(join_subtokens_for_tree, reference_trees)]
        if self.tree_format == 'slash':
            try:
                inputs_trees = [*map(slash_to_bracket, inputs_trees)]
                reference_trees = [*map(slash_to_bracket, reference_trees)]
            except:
                warnings.warn(f'error when converting slash ({inputs_trees}) to bracket, skip this sample')
                return {'bracket_f1': 0}
            
        f1s = [myevalb_sentence_score(input_tree, reference_tree, 'relaxed' )[2] for input_tree, reference_tree in zip(inputs_trees, reference_trees)]
        return {'bracket_f1': sum(f1s) / len(f1s)}


    
    def evaluate(self, inputs: List[List[str]], references: List[List[List[str]]], task_name: str):
        evaluation_method = getattr(self, f'evaluate_{task_name}')
        assert len(inputs) == len(references), f'error, length of inputs ({len(inputs)}) and references ({len(references)}) doesn\'t match'
        return evaluation_method(inputs, [reference[0] for reference in references])
    
    def compute_acc(self, inputs: List[List[str]], references: List[List[str]]):
        """calculate how many inputs totally match references (`True</s>` or `False</s>`) """
        num_matches = 0
        for input_seq, reference_seq in zip(inputs, references):
            if input_seq == reference_seq:
                num_matches += 1
        return {'acc': num_matches / len(inputs)}

    
    def evaluate_constituency_discrimination(self, inputs: List[List[str]], references: List[List[str]]):
        """constituency_discrimination's evaluation method (accuracy), """
        return self.compute_acc(inputs, references)
    
    def evaluate_sequence_f1(self, inputs: List[List[str]], references: List[List[str]]):
        total_p, total_r, total_f1 = 0, 0, 0
        for input_seq, reference_seq in zip(inputs, references):
            # pop "</s>"
            input_seq = input_seq[:-1]
            reference_seq = reference_seq[:-1]
            p, r, f1 = calculate_prf(input_seq, reference_seq)
            total_f1 += f1
            total_p += p
            total_r += r
        
        return {'p': total_p / len(inputs), 'r': total_r / len(inputs), 'f1': total_f1 / len(inputs)}

    
    def evaluate_span_f1(self, inputs: List[List[str]], references: List[List[str]], sep_token: str = '<sep>'):
        # calculate Precision, Recall and F1 between input spans and reference spans
        total_p, total_r, total_f1 = 0, 0, 0
        for input_seq, reference_seq in zip(inputs, references):
            # pop "</s>"
            input_seq = input_seq[:-1]
            reference_seq = reference_seq[:-1]
            # split by "<sep>" (get spans)
            input_spans = split_list(input_seq, sep_token)
            reference_spans = split_list(reference_seq, sep_token)

            p, r, f1 = calculate_prf(input_spans, reference_spans)
            total_f1 += f1
            total_p += p
            total_r += r
        
        return {'p': total_p / len(inputs), 'r': total_r / len(inputs), 'f1': total_f1 / len(inputs)}

    
    def evaluate_constituency_searching(self, inputs: List[List[str]], references: List[List[str]]):
        return {'f1': self.evaluate_span_f1(inputs, references)['f1']}
    
    # def evaluate_pos_tagging(self, inputs: List[List[str]], references: List[List[str]]):
    #     """calculate pos tagging accuracy"""
    #     for input_seq, reference_seq in zip(inputs, references):
    #         # pop "</s>"
    #         input_seq = input_seq[:-1]
    #         reference_seq = reference_seq[:-1]

    def evaluate_production_detection(self, inputs: List[List[str]], references: List[List[str]]):
        return {'f1': self.evaluate_span_f1(inputs, references)['f1']}
    
    def evaluate_height_selection(self, inputs: List[List[str]], references: List[List[str]]):
        return {'f1': self.evaluate_sequence_f1(inputs, references)['f1']}
    
    def evaluate_tree_pruning(self, inputs: List[List[str]], references: List[List[str]]):
        return self.evaluate_myevalb(inputs, references)

    def evaluate_pruned_tree_completion(self, inputs: List[List[str]], references: List[List[str]]):
        return self.evaluate_myevalb(inputs, references)

    def evaluate_pruned_tree_parse(self, inputs: List[List[str]], references: List[List[str]]):
        return self.evaluate_myevalb(inputs, references)
    
    def evaluate_treeposition_indexing(self, inputs: List[List[str]], references: List[List[str]]):
        return self.compute_acc(inputs, references)

    def evaluate_tree_forming(self, inputs: List[List[str]], references: List[List[str]]):
        return self.evaluate_myevalb(inputs, references)
    
    def evaluate_node_deleting(self, inputs: List[List[str]], references: List[List[str]]):
        return self.evaluate_myevalb(inputs, references)




def evaluation_and_calculate_bleu(output_dir: str, step_or_name: Union[int, str], eval_dataloader: DataLoader, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, device: Union[int, str]):
    """single-process version of `para_pretrain_multitask.ditraibuted_evaluation_and_calculate_bleu`"""
    # do evaluation
    model.eval()
    all_preds = []
    for batch in tqdm.tqdm(eval_dataloader, desc='evaluating...'):
        inputs, labels, metadata = batch
        inputs = inputs.to(device)
        res = model.generate(**inputs, max_new_tokens=128, output_scores=True, return_dict_in_generate=True)
        res_sequences = res.sequences
        print(*(tokenizer.convert_ids_to_tokens(filter(lambda x: x != tokenizer.pad_token_id,input_seq)) for input_seq in inputs.input_ids), sep='\n', file=open('../outputs/debug_outputs/val_instances_single', 'a'))
        # return
        for idx, seq in enumerate(res_sequences):
            # print(tokenizer.convert_ids_to_tokens(seq))
            seq = [each for each in seq if each != tokenizer.pad_token_id] # remove pad tokens of generated results
            # print(tokenizer.convert_ids_to_tokens(seq))
            if 'scpg' in metadata[idx]['task']:
                all_preds.append({ # use space seperated tokens as decoding method
                    'pred': ' '.join(tokenizer.convert_ids_to_tokens(seq)), 
                    'label': ' '.join(tokenizer.convert_ids_to_tokens(labels[idx])), 
                    **metadata[idx]
                })
            else:
                all_preds.append({ # use space seperated tokens as decoding method
                    'pred': ' '.join(tokenizer.convert_ids_to_tokens(seq)), 
                    'label': ' '.join(tokenizer.convert_ids_to_tokens(labels[idx])), 
                    **metadata[idx]
                })
            # if metadata[idx]['task'] == 'scpg':
            #     print('===' * 30)
            #     print(tokenizer.decode(inputs.input_ids[idx]).replace('<pad>', ''))
            #     print(tokenizer.decode(seq))

    if isinstance(step_or_name, int):
        results_dir = os.path.join(output_dir, 'results', f'{step_or_name:07d}')
    else:
        results_dir = os.path.join(output_dir, 'results', step_or_name)
    os.makedirs(results_dir, exist_ok=True)
    # all_data = []
    
    # split result instances according to task
    predictions = defaultdict(list)
    labels = defaultdict(list)
    for instance in all_preds:
        predictions[instance['task']].append(instance['pred'].split(' '))
        labels[instance['task']].append([instance['label'].split(' ')])
    
    for task_name in predictions:
        with open(os.path.join(results_dir, f'{task_name}.txt'), 'w') as f:
            for input_line, label_line in zip(predictions[task_name], labels[task_name]):
                f.write('=================================\n')
                f.write(''.join(input_line).replace(chr(0x2581), ' ') + '\n')
                f.write(''.join(label_line[0]).replace(chr(0x2581), ' ') + '\n')

    # calculate bleu score
    task_names = sorted(predictions.keys())
    print(f'=' * 80)
    print(f'TASK NAME\tBLEU SCORE')
    bleu_res_dict = {}
    for task_name in task_names:
        task_predictions = predictions[task_name]
        task_labels = labels[task_name]
        bleu_score = corpus_bleu(task_labels, task_predictions) * 100
        print(f"{task_name}\t{bleu_score:.2f}")
        bleu_res_dict[task_name] = bleu_score
    print(f'=' * 80)
    return bleu_res_dict

def do_validation(
        output_dir: str, step_or_name: Union[int, str],
        val_data_root_dir: str, 
        val_batch_size: int,
        tokenizer_path: str, model_path: str, device: Union[int, str]):
    print(f'loading dataset, tokenizer, and model')
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    dataset = ParaPretrainMultitaskValDataset(val_data_root_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=val_batch_size, num_workers=10, collate_fn=dataset.collate_fn)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    evaluation_and_calculate_bleu(output_dir=output_dir, step_or_name=step_or_name, eval_dataloader=dataloader, tokenizer=tokenizer, model=model, device=device)

# def format_metric(metric_dict: Dict[str, float], step: str):
#     metric_df = DataFrame(metric_dict)

def evaluate_step(step: str, run_dir: str, evaluator: ParaPretrainMultitaskEvaluator, summary_writer: Union[str, SummaryWriter] = None, silent: bool = False):
    if summary_writer == 'global':
        global sw
        summary_writer = sw
    results = []
    result_json_dirs = [each for each in os.listdir(os.path.join(run_dir, 'results', step)) if re.match(r'\d\d.json', each)]
    for results_json_dir in result_json_dirs:
        with open(os.path.join(run_dir, 'results', step, results_json_dir), 'r') as f:
            results += json.load(f)

    predictions = defaultdict(list)
    labels = defaultdict(list)
    for instance in results:
        predictions[instance['task']].append(instance['pred'].split(" "))
        labels[instance['task']].append([instance['label'].split(" ")])

    task_names = sorted(predictions.keys())
    if not silent:
        print(f'=' * 80)
        print(f'step {step}\t|\t{len(predictions)} tasks\t\t|\t{sum((map(len, predictions.values())))} samples')
        print(f'TASK NAME\t|\tSCORE NAME\t|\tSCORE VALUE')
        print(f'=' * 80)
    bleu_res_dict = {}
    for task_name in task_names:
        task_predictions = predictions[task_name]
        task_labels = labels[task_name]
        if task_name in evaluator.supported_tasks:
            score_name, score = [*evaluator.evaluate(task_predictions, task_labels, task_name).items()][0]
            score *= 100
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = corpus_bleu(task_labels, task_predictions) * 100
                score_name = 'bleu-4'
        if summary_writer is not None:
            summary_writer.add_scalar(f'{task_name}_{score_name}', score, int(step))
        if not silent:
            print(f"{task_name}\t|\t{score_name}\t|\t{score:.2f}")
        bleu_res_dict[task_name] = score
    
    if silent:
        return step, bleu_res_dict
    else:
        return bleu_res_dict

def do_evaluation(run_dir: str, step: Union[int, List[int], str] = 'all', min_step: int = None, tensorboard_log_dir: str = None, tokenizer_pretrained_path: str = '../pretrained-models/syntax-t5-base-node/', tree_format: Literal['slash', 'bracket'] = None, mp: bool = False):
    """
    do evaluation like what have been done in `para_pretrain_multitask.py`
    Args:
        run_dir: the directory of the run
        step: the step of the run to be evaluated, if 'all', evaluate all steps
        min_step: (optional) the minimum step to be evaluated, if None, evaluate all steps, will be overrided if `step` is set
        tensorboard_log_dir: (optional) the directory to save tensorboard logs after evaluation for better comparison
        tokenizer_pretrained_path: the path of the pretrained tokenizer
        tree_format: the format of the tree, 'slash' or 'bracket'
    """
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_pretrained_path)
    evaluator = ParaPretrainMultitaskEvaluator('../evaluation/apps/EVALB/evalb', '../evaluation/apps/EVALB/sample/sample.prm', tokenizer, tree_format)
    # if isinstance(run_dirs, str):
    #     run_dirs = [run_dirs]
    global sw
    if tensorboard_log_dir is not None:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        if os.listdir(tensorboard_log_dir):
            os.system(f'rm {tensorboard_log_dir}/events*')
        
        sw = SummaryWriter(tensorboard_log_dir)
    else:
        sw = None

    if min_step is None:
        min_step = -1
    if step == 'all':
        available_steps = [*filter(lambda x: re.match(r'\d{7}', x) and int(x) > min_step, os.listdir(os.path.join(run_dir, 'results')))]
    else:
        if isinstance(step, int):
            available_steps = [f'{step:07d}']
        else:
            available_steps = [f'{each:07d}' for each in step if each > min_step]
        for each in available_steps:
            assert os.path.exists(os.path.join(run_dir, 'results', each)), f'step `{each}` under {os.path.join(run_dir, "results")} not exist'
    if not mp:
        for step in available_steps:
            evaluate_step(step, run_dir, evaluator, sw)
    else:
        with Pool(processes=len(available_steps)) as p:
            results = p.starmap(evaluate_step, [(step, run_dir, evaluator, "global", True) for step in available_steps])
        for each in sorted(results, key=lambda x: int(x[0])):
            print('===' * 30)
            print(f'step: {each[0]}')
            print('===' * 30)
            print(each[1])
            

if __name__ == "__main__":
    # do_evaluation('./runs/tree_1_10_data_bracket/', step='all', tensorboard_log_dir='./.tb_compare/bracket', tokenizer_pretrained_path='../pretrained-models/syntax-t5-base-node/', tree_format='bracket', min_step=512)
    do_evaluation('./runs/tree_1_10_data_bracket/', step='all', tensorboard_log_dir="./.tb_compare/bracket_2", tokenizer_pretrained_path='../pretrained-models/syntax-t5-base-node', tree_format='bracket', min_step=1024)
    do_evaluation('./runs/tree_1_10_data_slash/', step='all', tensorboard_log_dir="./.tb_compare/slash_2", tokenizer_pretrained_path='../pretrained-models/syntax-t5-base-node', tree_format='slash', min_step=1024)
    # do_validation('./runs/random_test', '0106495_test', '../data/ParaNMT50m_original/val_data/', 256, '../pretrained-models/syntax-t5-base', './runs/random_test/checkpoints/model/', 'cpu')