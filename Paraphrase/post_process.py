from typing import List, Union, Literal, Any, Callable
import re
import pickle as pkl
import os
import sys; sys.path.append('..')
from evaluation.eval import main as eval_main
from evaluation.myevalb import sentence_score
from data.TreeUtils.tree import MyTree
import pandas as pd
from itertools import chain
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.data import path as nltk_path
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import json
from transformers.generation import GreedySearchDecoderOnlyOutput
from tempfile import TemporaryDirectory
import pickle as pkl

nltk_path.append(os.path.abspath('../nltk_data'))
# all NT tokens sorted by length (longest first)
NT_tokens = sorted(filter(lambda x: bool(x), pkl.load(open('./data/ParaNMT50m_original/NT.pkl', 'rb'))), key=lambda x: len(x), reverse=True)

def retokenize_files(run_dir: str, process_method: Callable = None):
    run_dir = os.path.join(run_dir, 'results', 'exemplar')
    available_epoch_results = [*filter(lambda fname: re.match(r'\d+\.txt', fname), os.listdir(run_dir))]
    print(f'found availabel files: {", ".join(available_epoch_results)} in `{run_dir}`')
    for file_name in available_epoch_results:
        print(f'processing {file_name}...')
        file_name = os.path.join(run_dir, file_name)
        file_contents = [each for each in open(file_name, 'r').read().split('\n') if each]
        for idx, sent in enumerate(file_contents):
            if process_method is not None:
                sent = process_method(sent)
            sent = ' '.join(nltk.word_tokenize(sent))
            file_contents[idx] = sent
        
        with open(file_name, 'w') as f:
            f.write('\n'.join(file_contents))


def get_parse_corpus_bleu(input_file_path: str, dataset_dir: str):
    src_trees = map(
        lambda x: x.prune_downwards(5, 1, inference_height=True).tostr_with_bracket_spaces(), 
        pkl.load(open(os.path.join(dataset_dir, 'src_trees.pkl'), 'rb'))
    )
    temp_trees = map(
        lambda x: x.prune_downwards(5, 1, inference_height=True).tostr_with_bracket_spaces(), 
        pkl.load(open(os.path.join(dataset_dir, 'ref_trees.pkl'), 'rb'))
    )
    references = [[nltk.word_tokenize(each)] for each in chain(*zip(src_trees, temp_trees))]
    inputs = [nltk.word_tokenize(each) for each in open(input_file_path, 'r')]
    return corpus_bleu(references, inputs) * 100



def get_tree_leaf_spans(tree_str: str):
    """
    step 1: get all terminal brackets (brackets that do not contain sub-brackets)
    step 2: strip out the subtree-bracket (the NT token following left bracket and return the rest)
    """
    all_terminal_brackets = re.findall(r'\([^()]*\)', tree_str)
    res = ''
    for bracket_content in all_terminal_brackets:
        bracket_content = bracket_content[1:-1].strip() # strip out brackets
        if ' ' in bracket_content:
            res += (' ' + bracket_content.split(' ', maxsplit=1)[1])
        else: # iterate over NT_tokens, find the longest match of NT_token as the subtree-root, append the string without subtree-root
            # matched = False
            for NT in NT_tokens: 
                if bracket_content.startswith(NT):
                    # matched = True
                    res += (' ' + bracket_content[len(NT):])
                    break
            
            # if not matched: # if none of them matches the span, regard the 1st char of the bracket content as subtree-root
            #     res += (' ' + bracket_content[1:])
    
    return res
    

def extract_target_extra_id(sent: str):
    extra_ids = re.findall(r'<extra_id_([0-9])+>', sent)
    extra_ids = sorted(map(int, extra_ids))
    res = ''
    for l_extra_id, r_extra_id in zip(extra_ids[:-1], extra_ids[1:]):
        ans_span = re.search(f'<extra_id_{l_extra_id}>(.*)<extra_id_{r_extra_id}>', sent)
        if ans_span:
            res += ans_span.group(1)
    
    return re.sub(r' +', ' ', res.strip())

def extract_target_after_prefix(sent: str, prefix: str):
    """extract target sentence after the last occurance of `prefix`"""
    prefix_start_index = sent.rfind(prefix)
    if prefix_start_index >= 0:
        return sent[prefix_start_index + len(prefix):].strip()
    else:
        return ''

def extract_target_before_prefix(sent: str, prefix: str):
    """extract target sentence before the first occurance of `prefix`"""
    prefix_start_index = sent.find(prefix)
    if prefix_start_index >= 0:
        return sent[:prefix_start_index].strip()
    else:
        return sent

def extract_target_triplet_tree(sent: str):
    tree_str = extract_target_after_prefix(sent, '<tgt>')
    tree_str = get_tree_leaf_spans(tree_str)
    return tree_str

def extract_target(sent: str, extract_method: str):
    """ extract target sentence (and probably other constituents) from `sent` according to `extract_method`, return a dict containing every constituent """
    sent = sent.strip().replace('</s>', '').replace('<s>', '')
    if extract_method == 'extra_id':
        sent = extract_target_extra_id(sent)
    elif extract_method == 'triplet_tree':
        sent = extract_target_triplet_tree(sent)
    elif extract_method == 'leaf':
        sent = get_tree_leaf_spans(sent)
    elif extract_method == 'direct':
        pass
    elif extract_method in ['sentence_tree', 'common_tree', 'sentence_tgt_tree']:
        if sent.find('<tgt>') >= 0: # if <tgt> is in the sentence, extract the tree after <tgt>
            tree = sent[sent.find('<tgt>') + len('<tgt>'):].replace('<tgt>', '').replace('[', '').replace(']', '')
        else:
            tree = ''
        sent = extract_target_before_prefix(sent, '<tgt>')
    
    # elif extract_method == 'sentence_tgt_tree':
    #     if sent.find('<sep>') >= 0: # if <tgt> is in the sentence, extract the tree after <tgt>
    #         tree = sent[sent.find('<sep>') + len('<sep>'):]
    #     else:
    #         tree = ''
    #     sent = extract_target_before_prefix(sent, '<sep>')


    elif extract_method.startswith('after_'):
        extra_method_match = re.match(r'after_`([^`]*)`', extract_method)
        assert extra_method_match, f"error, cannot extract prefix token from extract_method '{extract_method}'"
        prefix = extra_method_match.group(1)
        sent = extract_target_after_prefix(sent, prefix)
    elif extract_method.startswith('before_'):
        extra_method_match = re.match(r'before_`([^`]*)`', extract_method)
        assert extra_method_match, f"error, cannot extract prefix token from extract_method '{extract_method}'"
        prefix = extra_method_match.group(1)
        sent = extract_target_before_prefix(sent, prefix)
    else:
        raise NotImplementedError(f'error, extract method `{extract_method}` not supported')
    
    sent = sent.strip()
    res = {}
    if sent != '': # tokenize and split by space to stay consistent with reference files
        res['sent'] = ' '.join(nltk.word_tokenize(sent)) 
    else:
        res['sent'] = "-------------------------------------"
    
    if extract_method in ['sentence_tree', 'common_tree', 'sentence_tgt_tree']:
        res['tree'] = tree
    
    return res


def main(
    output_dir: str, 
    data_mode: Literal["target", "exemplar"], 
    dataset: str, 
    extract_method: str, 
    epoch: Union[int, str] = None, 
    num_chunks: int = None, 
    write_to_json_file: bool = True,
    skip_text_metrics: bool = False
):
    available_files = [] # (file_name, epoch, chunk_name)
    results_dir = os.path.join('./runs', output_dir, 'results', data_mode)
    if epoch is None or num_chunks is None:
        # find all files matching `[epoch]_chunk[chunk_idx]_raw.txt`
        for file_name in os.listdir(results_dir):
            if (file_name_match := re.match(r'([a-z0-9]+)_chunk([0-9]+)_raw.txt', file_name)):
                available_files.append((file_name, file_name_match.group(1), file_name_match.group(2)))

    # epoch = str(epoch)
    if epoch is not None:
        epochs = [epoch]
        # available_files = [*filter(lambda x: x[1] == epoch, available_files)]
    else: 
        # if epoch is not provided, extract it from matched file names 
        epochs = []
        for _, epoch, _ in available_files:
            if epoch not in epochs:
                epochs.append(epoch)
        epochs.sort(key=lambda x: (int(x) if x.isdigit() else 10_000))
    
    chunks = []
    if num_chunks is not None:
        chunks = [*range(num_chunks)]
        # available_files = [*filter(lambda x: int(x[2]) <= num_chunks, available_files)]
    else:
        # if `num_chunks` is not provided, filter out results corresponding to it
        for _, _, chunk_idx in available_files:
            chunk_idx = int(chunk_idx)
            if chunk_idx not in chunks:
                chunks.append(chunk_idx)
    
    chunks.sort()
    metrics_jsons = []

    if len(epochs) > 1:
        # fork `len(epochs) - 1` subprocesses to process each epoch
        ppid = os.getpid()
        for epoch in epochs[1:]:
            os.fork()
            if os.getpid() != ppid:
                break
        
        if os.getpid() == ppid:
            epoch = epochs[0]
    
        print(f'pid {os.getpid()} [PPID: {ppid}] is processing epoch {epoch}')

    epoch_result_sentences = []
    epoch_result_trees = []
    for chunk in chunks:
        with open(os.path.join(results_dir, f'{epoch}_chunk{chunk}_raw.txt'), 'r') as f:
            for line in f: # process each line
                # extract sentences and (optional) trees and append them to `epoch_result_sentences` and `epoch_result_trees`
                extracted_result = extract_target(line.strip(), extract_method)
                epoch_result_sentences.append(extracted_result['sent'])
                if extract_method in ['sentence_tree', 'common_tree', 'sentence_tgt_tree']:
                    if extract_method == 'common_tree':
                        epoch_result_trees.append(re.sub(r'\*( *)([^ ()]*)( *)\*', r'*\2*', extracted_result['tree']))
                    else:
                        epoch_result_trees.append(extracted_result['tree'])
    
    with open(os.path.join(results_dir, f'{epoch}.txt'), 'w') as f:
        f.write('\n'.join(epoch_result_sentences))
    
    # if extract_method == 'sentence_tree':
    #     with open(os.path.join(results_dir, f'{epoch}_tree.txt'), 'w') as f:
    #         f.write('\n'.join(epoch_result_trees))
    
    # try:
    os.chdir('./evaluation')
    if 'parse' not in output_dir:
        # call `main()` from `evaluation/eval.py`
        if skip_text_metrics:
            metrics_jsons.append({})
        else:
            metrics_jsons.append(
                eval_main(
                    os.path.join(os.path.join('../runs', output_dir, 'results', data_mode), f'{epoch}.txt'), 
                    os.path.join('..', dataset, 'tgt.txt'), 
                    os.path.join('..', dataset, 'ref.txt'),
                    write_to_metrics=write_to_json_file
                )
            )
        if extract_method in ['sentence_tree', 'common_tree', 'sentence_tgt_tree']:
            # if extract_method also extracts trees, calculate tree f1
            if extract_method in ['sentence_tree', 'common_tree',]:
                gt_ref_parses = [
                    (each['ref_tree'] if extract_method == 'sentence_tree' else each['common_tree']) \
                        for each in pkl.load(open(os.path.join('..', dataset, 'cached_data_t5_p5'), 'rb'))
                    ]
            else:
                gt_ref_parses = [ each['tgt_tree'] for each in pkl.load(open(os.path.join('..', dataset, 'cached_data_t5_p5'), 'rb')) ]
            epoch_f1 = [sentence_score(line, gold_line, 'relaxed')[-1] for line, gold_line in zip(epoch_result_trees, gt_ref_parses)]
            avrg_f1 = sum(epoch_f1) / len(epoch_f1)
            print(f'tree f1: {avrg_f1}')
            # metrics_jsons[-1].update({'f1': avrg_f1 * 100})
            metrics_jsons[-1].update({'f1': avrg_f1 * 100})
            json_file_content = json.load(open(os.path.join('../runs', output_dir, 'results', data_mode, 'metrics.json'), 'r'))
            json_file_content[str(epoch)].update({'f1': avrg_f1 * 100})
            json.dump(json_file_content, open(os.path.join('../runs', output_dir, 'results', data_mode, 'metrics.json'), 'w'), indent=2)
    else:
        metrics_jsons.append({'bleu': get_parse_corpus_bleu(
            os.path.join(os.path.join('../runs', output_dir, 'results', data_mode), f'{epoch}.txt'),
            '.' + dataset
            )})
    os.chdir('..')
    # except AssertionError as e:

    return pd.DataFrame(metrics_jsons)

def write_metrics_to_tensorboard(metrics_df: pd.DataFrame, epoch: int, mode: Literal['exemplar', 'target'], sw: SummaryWriter):
    for metric_name in metrics_df:
        metric_val = metrics_df[metric_name].item()
        if metric_val >= 0:
            sw.add_scalar(f'{metric_name}/{mode}', metric_val, epoch)
        
def subset_process(
    output_dir: str, 
    data_mode: Literal["target", "exemplar"], 
    dataset: str, 
    extract_method: str, 
    sentence_mask: List[int], # 1 stands for choosen
    epoch: Union[int, str] = None, 
    num_chunks: int = 1, 
    write_results_json: bool = False,
    silent: bool = False
):
    prev_stdout, prev_stderr = sys.stdout, sys.stderr
    if silent:
        sys.stdout, sys.stderr = None, None
    
    print(f'running epoch {epoch} in silent mode (write_results_json = {write_results_json})', file=prev_stdout)

    def write_dataset_subset(dataset_filepath, mask, out_dir):
        _, dataset_filename = os.path.split(dataset_filepath)
        res = []
        with open(dataset_filepath, 'r') as f:
            for idx, line in enumerate(f.read().split('\n')):
                if idx < len(mask) and mask[idx]:
                    res.append(line)
        
        with open((out_fname := os.path.join(out_dir, dataset_filename)), 'w') as f:
            f.write('\n'.join(res))
        
        return out_fname
    
    def write_dataset_tree_subset(tree_filepath, mask, out_dir):
        _, tree_filename = os.path.split(tree_filepath)
        res = []
        with open(tree_filepath, 'rb') as f:
            for idx, tree in enumerate(pkl.load(f)):
                if idx < len(mask) and mask[idx]:
                    res.append(tree)

        
        with open((out_fname := os.path.join(out_dir, tree_filename)), 'wb') as f:
            pkl.dump(res, f)

        return out_fname 

    with TemporaryDirectory() as temp_d:
        print(f'writing to `{temp_d}` (deleted upon run end)')
        results_dir = os.path.join(output_dir, 'results', data_mode)

        if epoch is None:
            # calculate epoch with best overall performance
            with open(os.path.join(results_dir, 'metrics.json')) as f:
                metrics = json.load(f)
                best_metric_sum, best_epoch = -100, 0
                for epoch, metric_name2val in metrics.items():
                    metric_val = [val if 'TED' not in name else -val for (name, val) in metric_name2val.items()]
                    metric_sum = sum(metric_val)
                    if metric_sum > best_metric_sum:
                        best_metric_sum = metric_sum
                        best_epoch = epoch
        
            epoch = best_epoch

        epoch_result_sentences = []
        for chunk in range(num_chunks):
            with open(os.path.join(results_dir, f'{epoch}_chunk{chunk}_raw.txt'), 'r') as f:
                for idx, line in enumerate(f): # process each line
                    # extract sentences and (optional) trees and append them to `epoch_result_sentences` and `epoch_result_trees`
                    if sentence_mask[idx]:
                        extracted_result = extract_target(line.strip(), extract_method)
                        epoch_result_sentences.append(extracted_result['sent'])
                        # if extract_method in ['sentence_tree', 'common_tree', 'sentence_tgt_tree']:
                        #     if extract_method == 'common_tree':
                        #         epoch_result_trees.append(re.sub(r'\*( *)([^ ()]*)( *)\*', r'*\2*', extracted_result['tree']))
                        #     else:
                        #         epoch_result_trees.append(extracted_result['tree'])
        
        with open((res_newpth := os.path.join(temp_d, f'{epoch}.txt')), 'w') as f:
            f.write('\n'.join(epoch_result_sentences))
        
        # if extract_method == 'sentence_tree':
        #     with open(os.path.join(results_dir, f'{epoch}_tree.txt'), 'w') as f:
        #         f.write('\n'.join(epoch_result_trees))
        
        # try:
        tgt_newpth = write_dataset_subset(os.path.join(dataset, 'tgt.txt'), sentence_mask, temp_d)
        ref_newpth = write_dataset_subset(os.path.join(dataset, 'ref.txt'), sentence_mask, temp_d)
        write_dataset_tree_subset(os.path.join(dataset, 'tgt_trees_for_evaluation.pkl'), sentence_mask, temp_d)
        write_dataset_tree_subset(os.path.join(dataset, 'ref_trees_for_evaluation.pkl'), sentence_mask, temp_d)
        os.chdir('./evaluation')
        # call `main()` from `evaluation/eval.py`
        print(f'epoch {epoch} start processing...', file=prev_stdout)
        metrics_json = eval_main(
            res_newpth, 
            tgt_newpth, 
            ref_newpth,
            write_to_metrics=False
        )
        os.chdir('..')

        print(f'process complede!', file=prev_stdout)
        if write_results_json:
            results_json_path = os.path.join(results_dir, f'metrics_long_{sum(sentence_mask)}.json')
            print(f'writing to result json dir {results_json_path}', file=prev_stdout)
            if not os.path.exists(results_json_path):
                with open(results_json_path, 'w') as f:
                    json.dump({str(epoch): metrics_json}, f)
            else:
                prev_json = json.load(open(results_json_path, 'r'))
                prev_json.update({str(epoch): metrics_json})
                with open(results_json_path, 'w') as f:
                    json.dump(prev_json, f)


        if silent:
            sys.stdout, sys.stderr = prev_stdout, prev_stderr
        return metrics_json


if __name__ == '__main__':
    assert len(sys.argv) > 1, f'error, subcommand not specified'
    print(sys.argv)
    subcommand = sys.argv.pop(1)
    parser = ArgumentParser()
    if subcommand == 'main':
        parser.add_argument('-e', '--epoch', type=int)
        args = parser.parse_args()
        # print(args.epoch)
        main('para_train_t5/_stage2/triplet_sentencetree_fromscratch_3',
            'exemplar',
            './data/ParaNMT50m_triple/test',
            'sentence_tree',
            args.epoch,
            write_to_json_file=False,
            skip_text_metrics=True,
        )
        # main('para_train_t5/_stage2/triplet_sentencetgttree_finetune_3', 'exemplar', './data/ParaNMT50m_triple/test', 'sentence_tgt_tree', args.epoch)
        # main('para_train_t5/_triplet/triplet_sentencetree_finetune_2', 'exemplar', './data/ParaNMT50m_triple/test', 'sentence_tree', args.epoch)
    elif subcommand == 'retokenize':
        parser.add_argument('-d', '--run-dir', type=str)
        args = parser.parse_args()
        strip_prefix = lambda x: re.sub(r'^target sentence :', '', x)
        run_dir = args.run_dir or './runs/para_train_t5/_triplet/triplet_notree_finetune_qqppos/'
        retokenize_files(run_dir, strip_prefix if 'notree' in run_dir and ('_2' not in run_dir) else None)
    elif subcommand == 'subset':
        with open('./pickles/data_statistics/long_sentence_mask/ParaNMT50m_50.pkl', 'rb') as f:
            sentence_mask = pkl.load(f)
        subset_process("./runs/para_train_t5/_stage2/triplet_sentencetree_fromscratch_3/", 'exemplar', 'data/ParaNMT50m/test', 'sentence_tree', sentence_mask)
        subset_process("./runs/para_train_t5/_stage2/triplet_sentencetree_full_inetune_3/", 'exemplar', 'data/ParaNMT50m/test', 'sentence_tree', sentence_mask)
    else:
        raise AttributeError(f'subcommand {subcommand} not recognized')

    # main('para_train_t5/triplet_tree_finetune', 'exemplar', './data/ParaNMT50m_triple/test', 'triplet_tree', 1, 1)
    # os.chdir('..')
    # get_parse_corpus_bleu('../runs/para_train_t5/triplet_parse_from_scratch/results/exemplar/1.txt', '../data/ParaNMT50m_triplet/test')
    # =====================`get_tree_leaf_spans`======================================
    # with open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/liuhongxu02/PromptTuning/runs/para_train_t5/triplet_tree_fromscratch/results/exemplar/test_chunk0_raw.txt', 'r') as f:
    #     for _, line in zip(range(20), f):
    #         line = line.strip().replace('</s>', '')
    #         if (tgt_index := line.find('<tgt>')) >= 0:
    #             print('original:')
    #             print(line[tgt_index + 5:])
    #             print('extracted:')
    #             print(get_tree_leaf_spans(line[tgt_index + 5:]))
