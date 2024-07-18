import os
import sys; 
if os.path.abspath('..') not in sys.path:
    sys.path.append(os.path.abspath('..'))
from Paraphrase.para_pretrain_dataset import ParaPretrainDataset, ParaPretrainIterableDataset
try:
    from .para_pretrain_multitask_dataset import ParaPretrainMultitaskCurriculumDataset, ParaPretrainMultitaskValDataset
    from .para_pretrain_evaluation_utils import ParaPretrainMultitaskEvaluator
except ImportError:
    from para_pretrain_multitask_dataset import ParaPretrainMultitaskCurriculumDataset, ParaPretrainMultitaskValDataset
    from para_pretrain_evaluation_utils import ParaPretrainMultitaskEvaluator
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from jsonargparse import ArgumentParser, ActionConfigFile, Namespace
from transformers import (
    PreTrainedModel, PreTrainedTokenizer,
    T5TokenizerFast, T5Tokenizer, T5ForConditionalGeneration,
    BartTokenizerFast, BartTokenizer, BartForConditionalGeneration,
    GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel,
    BertTokenizerFast, BertTokenizer, BertForMaskedLM,
    get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
)
from nltk.translate.bleu_score import corpus_bleu
from typing import List
import torch
from torch import nn, tensor, Tensor
from torch.utils.data import DataLoader, Dataset, Subset
import pickle as pkl
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
import logging 
from datetime import datetime, timedelta
from time import sleep
import re
from collections import defaultdict
import json

def print_rank_0(*args, **kwargs):
    if ('LOCAL_RANK' not in os.environ) or os.environ['LOCAL_RANK'] == '0':
        print(*args, **kwargs)

model_type_to_cls = {
    't5': (T5TokenizerFast, T5ForConditionalGeneration),
    'bart': (BartTokenizerFast, BartForConditionalGeneration),
    'gpt2': (GPT2TokenizerFast, GPT2LMHeadModel),
    'bert': (BertTokenizerFast, BertForMaskedLM)
}

def join_subtokens(tokens: List[str]):
    """return a new sequence with subtokens (tokens not starting with `U+2581`), this is suitable for natural language tasks (for example, scpg)"""
    new_seq = []
    this_token = ''
    for token in tokens:
        # print(token)
        if token.startswith(chr(0x2581)):
            if this_token:
                new_seq.append(this_token)
            this_token = token
        else:
            this_token += token
    
    new_seq.append(this_token)
    return new_seq



def clean_generated_sequence(generated_tokens: List[str]):
    """returns a cleaned version of generated sequence (replacing redundant `U+2581`s, surrounding syntax tree tokens with spaces, forming entire words in SentencePiece manner)"""
    pass

def update_run_state(run_state_path: str, **update_dict):
    previous_run_state = json.load(open(run_state_path, 'r')) if os.path.exists(run_state_path) else {}
    for key in update_dict:
        # assert key in previous_run_state, f"error, invalid run_state term `{key}`"
        previous_run_state[key] = update_dict[key]
    
    json.dump(previous_run_state, open(run_state_path, 'w'))

def distributed_evaluation_and_calculate_score(args: Namespace, step: int, acc: Accelerator, eval_dataloader: DataLoader, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration, evaluator: ParaPretrainMultitaskEvaluator):
    # do evaluation
    # model_pt = T5ForConditionalGeneration.from_pretrained('./runs/random_test/checkpoints/model').to(acc.local_process_index)
    model.eval()
    # model = model.cpu()
    # for model_pt_param, model_param in zip(model_pt.parameters(), model.parameters()):
    #     print((model_pt_param == model_param).all())
    all_preds = []
    # val_dataset = ParaPretrainMultitaskValDataset('../data/ParaNMT50m_original/val_data', tokenizer)
    # val_subset = Subset(val_dataset, range(0, len(val_dataset) // 2) if acc.is_main_process else range(len(val_dataset) // 2, len(val_dataset)))
    # eval_dataloader = DataLoader(val_subset, batch_size=args.val_batch_size // 2, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, collate_fn=val_dataset.collate_fn)
    for batch in tqdm(eval_dataloader, desc='evaluating...', ncols=100):
        inputs, labels, metadata = batch
        # print_rank_0(*(tokenizer.convert_ids_to_tokens(filter(lambda x: x != tokenizer.pad_token_id, input_seq)) for input_seq in inputs.input_ids), sep='\n', file=open('../outputs/debug_outputs/val_instances_multiple', 'a'))
        # inputs = inputs.to(acc.local_process_index)
        # inputs = inputs.to('cpu')
        res = model.generate(**inputs, max_new_tokens=256, output_scores=True, return_dict_in_generate=True)
        res_sequences = res.sequences.tolist()
        for idx, seq in enumerate(res_sequences):
            # print(tokenizer.convert_ids_to_tokens(seq))
            seq = [each for each in seq if each != tokenizer.pad_token_id] # remove pad tokens of generated results
            # filter out <extra_id>s for scpg task
            if metadata[idx]['task'] == 'scpg':
                seq = [*filter(lambda x: not (32000 <= x <= 32099), seq)]
            decoded_pred = tokenizer.convert_ids_to_tokens(seq)
            decoded_label = tokenizer.convert_ids_to_tokens(labels[idx])
            decoded_inputs = tokenizer.convert_ids_to_tokens([*filter(lambda x: x != tokenizer.pad_token_id, inputs.input_ids[idx].tolist())])
            if metadata[idx]['task'] == 'scpg':
                seq = [*filter(lambda x: not (32000 <= x <= 32099), seq)]
                decoded_pred, decoded_label = join_subtokens(decoded_pred), join_subtokens(decoded_label)
            # print(tokenizer.convert_ids_to_tokens(seq))
            all_preds.append({ # use space seperated tokens as decoding method
                'pred': ' '.join(decoded_pred), 
                'label': ' '.join(decoded_label), 
                'inputs': ' '.join(decoded_inputs),
                **metadata[idx]
            })
            # if metadata[idx]['task'] == 'scpg':
            #     print('===' * 30)
            #     print(tokenizer.decode(inputs.input_ids[idx]).replace('<pad>', ''))
            #     print(tokenizer.decode(seq))

    # serialize outputs as json
    if args.finetune:
        results_dir = os.path.join(args.output_dir, 'results', f'{"_".join(args.val_tasks)}', f'{step:02d}')
    else:
        results_dir = os.path.join(args.output_dir, 'results', f'{step:07d}_test' if args.evaluation_only else f'{step:07d}')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f'{acc.process_index:02d}.json'), 'w') as f:
        json.dump(all_preds, f, indent=2)
    
    acc.wait_for_everyone()

    # ===============================================================
    # calculate bleu score for each task within a directory consisting of generated results of different processes
    #  (a set of `xx.json` files)
    # ===============================================================
    # iterate over all processes' results json and retrieve results from all processes
    all_data = []
    for file_name in [each for each in os.listdir(results_dir) if re.match(r'\d\d.json', each)]:
        all_data.extend(json.load(open(os.path.join(results_dir, file_name))))
    
    # split result instances according to task
    inputs = defaultdict(list) # Dict[str, List[List[int]]]
    predictions = defaultdict(list) # Dict[str, List[List[int]]]
    labels = defaultdict(list) # Dict[str, List[List[List[int]]]]
    for instance in all_data:
        predictions[instance['task']].append(instance['pred'].split(" "))
        labels[instance['task']].append([instance['label'].split(" ")])
        inputs[instance['task']].append(instance['inputs'].split(" "))
    
    if acc.is_main_process:
        # export to seperated txt file results dir
        for task_name in predictions:
            with open(os.path.join(results_dir, f'{task_name}.txt'), 'w') as f:
                for input_line, pred_line, label_line in zip(inputs[task_name], predictions[task_name], labels[task_name]):
                    f.write('==========================================================================================\n')
                    f.write(''.join(input_line).replace(chr(0x2581), ' ') + '\n')
                    f.write(''.join(pred_line).replace(chr(0x2581), ' ') + '\n')
                    f.write(''.join(label_line[0]).replace(chr(0x2581), ' ') + '\n')

    # calculate bleu score
    task_names = sorted(predictions.keys())
    print_rank_0(f'=' * 50)
    print_rank_0(f"{'TASK NAME': ^30}|{'SCORE': ^8}|{'METRIC NAME': ^8}")
    print_rank_0(f'=' * 50)
    res_dict = {}
    logging.info(f'evaluation result at step {step}:')
    for task_name in task_names:
        task_predictions = predictions[task_name]
        task_labels = labels[task_name]
        if task_name in evaluator.supported_tasks:
            score_dict = evaluator.evaluate(task_predictions, task_labels, task_name)
            score_name, score_value = [*score_dict.items()][0]
            score_value *= 100
        else:
            score_name = 'bleu'
            score_value = corpus_bleu(task_labels, task_predictions) * 100
        print_rank_0(f"{task_name: ^30}|{score_value: ^8.2f}|{score_name: ^8}")
        logging.info(f"{task_name = }, {score_name = }, {score_value = }")
        res_dict[task_name] = score_name, score_value
    print_rank_0(f'=' * 50)
    return res_dict

def train(acc: Accelerator, args, dataloader: DataLoader, val_dataloader: DataLoader, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, evaluator: ParaPretrainMultitaskEvaluator):
    total_data_cnt = len(dataloader.dataset)
    # WARNING: to avoid counting total num_data for multiple times, this number is pre-computed and given here
    # but the accurate number might change if you modify the data
    # initialize optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_train_iters_per_epoch = total_data_cnt // args.batch_size + 1
    total_training_steps = num_train_iters_per_epoch * args.num_epochs // args.gradient_accumulation_steps
    num_warmup_steps = args.num_warmup_steps // args.gradient_accumulation_steps
    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps,) if 'all_data' not in args.output_dir else get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_training_steps)
    # prepare modules
    model, dataloader, val_dataloader, optimizer, lr_scheduler = \
        acc.prepare(model, dataloader, val_dataloader, optimizer, lr_scheduler)
    

    run_state_path = os.path.join(args.output_dir, 'checkpoints/training_state/run_state.json', )
    if args.resume:
        logging.info(f"resuming state from directory {os.path.join(args.output_dir, 'checkpoints/training_state/')}")
        acc.load_state(os.path.join(args.output_dir, 'checkpoints/training_state/'))
        run_state = json.load(open(run_state_path))

    tb_log_dir = args.output_dir if not args.finetune else os.path.join(args.output_dir, 'results', f'{"_".join(args.val_tasks)}')
    if (args.finetune or not args.resume) and acc.is_main_process: # when `finetune` is set to True, we always delete the tensorboard log files
        logging.info('not resuming, deleting tensorboard log files...')
        for fname in os.listdir(tb_log_dir):
            if 'events.out.tfevents' in fname:
                logging.info(f'deleting `{fname}`...')
                os.system(f'rm {tb_log_dir}/{fname}')
    if not args.evaluation_only and acc.is_main_process:
        sw = SummaryWriter(tb_log_dir, flush_secs=10)

    # initialize training metrics
    ema_loss = 0 #if not args.resume else run_state['best_loss']
    best_loss = 1e9 #if not args.resume else run_state['best_loss']
    best_score = 0 #if not args.resume else run_state['best_score']
    if args.resume and 'best_loss' in run_state:
        ema_loss = run_state['best_loss']
        best_loss = run_state['best_loss']
    if args.resume and 'best_score' in run_state:
        best_score = run_state['best_score']
    # total_loss = 0.0
    # current_train_iter = 0  # `current_train_iter` is used to calculate average loss
    # [optional] start at checkpoint epoch
    epoch_range = range(run_state['current_epoch'], args.num_epochs) if args.resume else range(args.num_epochs)
    if args.evaluation_only:
        global_step = run_state['current_epoch'] * len(dataloader) + run_state['current_iteration']
        logging.info(f'doing standalone evaluating at step {global_step}')
        distributed_evaluation_and_calculate_score(args, global_step, acc, val_dataloader, tokenizer, acc.unwrap_model(model), evaluator)
    else:
        for epoch in epoch_range:
            if acc.is_main_process:
                print('==' * 40)
                print(f'Training on Epoch ({epoch + 1}/{args.num_epochs})...')
                print('==' * 40)

            tqdm_train_loader = tqdm(dataloader, total=num_train_iters_per_epoch, ncols=100) if acc.is_local_main_process else dataloader
            if not args.finetune and args.resume and epoch == run_state['current_epoch']:
                logging.info(f'skipping dataloader (totally {run_state["current_iteration"] - 1} batches)...')
                # tqdm_train_loader = pkl.load(open(os.path.join(args.output_dir, f'checkpoints/training_state/tqdm_train_loader_{acc.process_index}.pkl'), 'rb'))

            for batch_idx, batch in enumerate(tqdm_train_loader):
                # if acc.local_process_index == 0:
                #     print(tokenizer.decode(batch.input_ids[0], skip_special_tokens=True))
                # this is code for testing block disposition
                if args.test_resume and (((not args.resume) and batch_idx == args.checkpoint_every + 1) or (args.resume and batch_idx == run_state['current_iteration'] + args.checkpoint_every + 1)): 
                    if acc.process_index == 0:
                        print('halting...')
                        sleep(35)
                        acc.wait_for_everyone()
                        print('halt resumed')
                    else:
                        acc.wait_for_everyone()

                # if args.resume and (not (args.num_epochs == 1 and args.skip_data_in_dataset)) and epoch == run_state['current_epoch'] and batch_idx < run_state["current_iteration"]:
                if not args.finetune and args.resume and epoch == run_state['current_epoch'] and batch_idx < run_state["current_iteration"]:
                    # skip data batches at checkpoint epoch
                    logging.debug(f'batch {batch_idx} skipped')
                    continue
                with acc.accumulate(model):
                    # breakpoint()
                    res = model(**batch)
                    loss = res.loss
                    optimizer.zero_grad()
                    acc.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    scalar_loss = res.loss.mean().item()
                    if acc.is_local_main_process:
                        tqdm_train_loader.set_postfix({
                            "loss": scalar_loss,
                            "lr": optimizer.param_groups[0]['lr']
                        })
                    if acc.is_main_process:
                        sw.add_scalar('loss', scalar_loss, epoch * len(dataloader) + batch_idx)
                    ema_loss = (ema_loss * 0.9 + scalar_loss) if ema_loss != 0 else scalar_loss
                    # accumulate and average loss
                    # total_loss += scalar_loss
                    # current_train_iter += 1
                if not args.finetune and (epoch * len(dataloader) + batch_idx) % args.evaluate_every == 0 and not (epoch == 0 and batch_idx == 0):  # evaluation every several steps
                    global_step = epoch * len(dataloader) + batch_idx
                    task2score = distributed_evaluation_and_calculate_score(
                        args,
                        global_step,
                        acc,
                        val_dataloader,
                        tokenizer,
                        acc.unwrap_model(model),
                        evaluator
                    )
                    for task_name in task2score:
                        score_name, score_value = task2score[task_name]
                        if acc.is_main_process:
                            sw.add_scalar(f'{score_name}_{task_name}', score_value, global_step)
                    scores = [each[1] for each in task2score.values()]
                    average_score = sum(scores) / len(scores)
                    # if total_loss / current_train_iter < best_loss:
                    if ema_loss < best_loss:
                        # logging.info(f'best loss: {total_loss / current_train_iter:.4f} (prev best: {best_loss:.4f})')
                        logging.info(f'best loss: {ema_loss:.4f} (prev best: {best_loss:.4f})')
                        best_loss = ema_loss
                        if acc.is_main_process: update_run_state(run_state_path, best_loss=best_loss)
                    if args.save_all_evaluate_checkpoints:
                        eval_ckpt_path = os.path.join(args.output_dir, 'checkpoints', 'every_model', f'{epoch}_{batch_idx}')
                        os.makedirs(eval_ckpt_path, exist_ok=True)
                        logging.info(f'saving model checkpoint to `{eval_ckpt_path}`...')
                        acc.unwrap_model(model).save_pretrained(eval_ckpt_path)
                    if average_score > best_score:
                        logging.info(f'saving model checkpoint at epoch {epoch} step {batch_idx} with '
                                    f'score {average_score:.2f} (prev best score {best_score:.2f})')
                        best_score = average_score
                        if acc.is_main_process:
                            # save best-performant model
                            acc.unwrap_model(model).save_pretrained(os.path.join(args.output_dir, 'checkpoints', 'model'))
                            update_run_state(run_state_path, best_score=best_score)
                            # json.dump({'best_loss': best_loss, 'best_score': best_score, 'current_epoch': epoch, 'current_iteration': batch_idx},
                            #     open(os.path.join(args.output_dir, 'checkpoints/training_state/run_state.json', ), 'w'))
                            
                        # acc.wait_for_everyone()
                        # pkl.dump(tqdm_train_loader, open(os.path.join(args.output_dir, f'checkpoints/training_state/tqdm_train_loader_{acc.process_index}.pkl'), 'wb'))
                        
                    else:
                        logging.info(f'not saving model checkpoint at step {len(dataloader) * epoch + batch_idx} since score {average_score:.4f} is more than prev best score {best_score:.4f})')
                    # acc.save_state(os.path.join(args.output_dir, 'checkpoints/training_state/'))

                if not args.finetune and (epoch * len(dataloader) + batch_idx) % args.checkpoint_every == 0 and not (batch_idx == 0 and epoch == 0): 
                    # there we save training (not model) checkpoints
                    # NOTE: we save model checkpoint ONLY if validation score is higher
                    # but we save training checkpoints (including model wieghts, optimizer states, scheduler states and random states)
                    # every `args.checkpoint_every` steps for resuming training
                    logging.info(f'saveing training state at epoch {epoch} step {batch_idx}...')
                    acc.save_state(os.path.join(args.output_dir, 'checkpoints/training_state'))
                    if acc.is_main_process:
                        update_run_state(run_state_path, 
                            current_epoch=epoch, current_iteration=batch_idx + 1) 
                        # `current_iteration` stands for the batch_idx we want to continue train on
            

            # evaluate at training end
            if epoch == args.num_epochs - 1 or args.finetune:
                global_step = epoch * len(dataloader) + batch_idx
                task2score = distributed_evaluation_and_calculate_score(
                    args,
                    global_step if not args.finetune else epoch,
                    acc,
                    val_dataloader,
                    tokenizer,
                    acc.unwrap_model(model),
                    evaluator
                )
                for task_name in task2score:
                    score_name, score_value = task2score[task_name]
                    if acc.is_main_process:
                        sw.add_scalar(f'{score_name}_{task_name}', score_value, global_step if not args.finetune else epoch)
                if args.finetune:
                    logging.info(f'saveing training state at the end of epoch {epoch}...')
                    acc.save_state(os.path.join(args.output_dir, 'checkpoints/training_state'))
                    if acc.is_main_process:
                        update_run_state(run_state_path, 
                            current_epoch=epoch + 1, current_iteration=0) 

    

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', action=ActionConfigFile)
    # parser.add_argument('-d', '--data-dir', type=str, default='data/ParaNMT50m_original/')
    # parser.add_argument('-M', '--model-type', type=str, default='t5')
    parser.add_class_arguments(ParaPretrainMultitaskCurriculumDataset, 'dataset', skip=['tokenizer', 'start_idx'])
    parser.add_argument('--use_cached_data', action='store_true', 
        help='whether or not use cached train h5 data and index file in `/tmp/para_pretrain_data/`,'
          'this might be beneficial if your training h5 data file is stored in a remote file storage cluster like hdfs,'
          ' while setting a large `num_workers` has similar outcomes')
    parser.add_argument('--skip_data_in_dataset', action='store_true',
        help='whether or not instantiate a truncted dataset if resuming from training, '
        'NOTE that this is only useful when `num_epochs` equals 1, '
        'and in order to make experiment reproduceable, we do not recommend using this option, since dataset might involve random actions (e.g. random shuffling if you set `shuffle=True`)')
    parser.add_argument('--force_recache', action='store_true', 
        help='whether or not ')
    parser.add_argument('--val_data_dir', type=str, help='directory storing validation data json files')
    parser.add_argument('--val_tasks', type=List[str], help='tasks included in validation', default=None)
    parser.add_class_arguments(ParaPretrainMultitaskEvaluator, 'evaluator', skip=['tokenizer'])
    # model args
    parser.add_argument('-s', '--seed', type=int, default=114514)
    parser.add_argument('-m', '--model_name_or_path', type=str, default='pretrained-models/t5-base/')
    parser.add_argument('-t', '--tokenizer_name_or_path', type=str, default='pretrained-models/syntax-t5-base/')
    # training and dataloade args
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('-e', '--num_epochs', type=int, default=5)
    parser.add_argument('-W', '--num_workers', type=int)
    parser.add_argument('-P', '--prefetch_factor', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('-g', '--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--half_batch_size', action='store_true', help='whether or not half the `batch_size` and double `gradient_accumulation_steps` (on machines with smaller)')
    parser.add_argument('--num_warmup_steps', type=int, default=1000)
    # checkpointing args
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('--checkpoint_every', type=int, default=1024, help='save checkpoint every `checkpoint_every` iterations')
    parser.add_argument('--evaluate_every', type=int, default=8192, help='proceed evaluation and metric calculation every `evaluate_every` iterations')
    parser.add_argument('--save_all_evaluate_checkpoints', action='store_true', help='whether or not save every single evaluation phase\'s corresponding checkpoint')
    parser.add_argument('-r', '--resume', action='store_true', help='whether or not resume training from checkpoint (checkpoint location is at ```output_dir```/checkpoints/training_state/)')
    parser.add_argument('--test-resume', action='store_true', help='whether or not test resuming')
    parser.add_argument('--finetune', action='store_true', help='whether this run is a finetuning run (single task, do not evaluate during training)')
    # parser.add_argument('--device', default=0)
    # parser.add_argument('--num_processes')
    # parser.add_class_arguments(Accelerator, 'accelerator',)
    # logging args
    parser.add_argument('--evaluation_only', action='store_true', help='whether or not only do evaluation')
    parser.add_argument('--logging.format', type=str, default='%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s')
    parser.add_argument('--logging.level', type=int, # choices=[logging.DEBUG, logging.INFO, logging.WARNING], 
        default=logging.INFO, help=f'{logging.DEBUG} for DEBUG, {logging.INFO} for INFO, {logging.WARNING} for WARNING, {logging.ERROR} for ERROR, {logging.CRITICAL} for CRITICAL')


    args = parser.parse_args()
    set_seed(args.seed)
    if args.half_batch_size:
        args.batch_size //= 2
        args.val_batch_size //= 2
        args.gradient_accumulation_steps *= 2
        args.checkpoint_every *= 2
        args.evaluate_every *= 2
    if args.finetune:
        args.output_dir = os.path.join(args.output_dir, 'finetune')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints', 'training_state'), exist_ok=True)
    if args.finetune: # no checkpoint, seperate results directory
        os.makedirs(os.path.join(args.output_dir, 'results', f'{"_".join(args.val_tasks)}'), exist_ok=True)
    else: 
        # os.makedirs(os.path.join(args.output_dir, 'checkpoints', 'training_state'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints', 'model'), exist_ok=True)
    logging.basicConfig(format=args.logging.format, level=args.logging.level, handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.output_dir, 'run.log'), 'w')])
    logging.info(f'setting seed `{args.seed}`')
    # make output directory and save config
    if args.evaluation_only:
        assert args.resume, f'Error, `evaluation-only` switch requires `resume` to be alse set as `True`'
    if args.resume and (not os.listdir(os.path.join(args.output_dir, 'checkpoints', 'training_state'))):
        logging.info(f'`resume` argument was set to `True` but there is no suitable checkpoints, overriding it to `False`')
        args.resume = False
    # os.makedirs(os.path.join(args.output_dir, 'checkpoints', 'dataloader'), exist_ok=True)

    parser.save(args, os.path.join(args.output_dir, 'config.yaml'), overwrite=True)
    # accelerator and logging configs

    distributed_timeout = timedelta(seconds=(30 if args.test_resume else 300))
    acc = Accelerator(device_placement=True, gradient_accumulation_steps=args.gradient_accumulation_steps, split_batches=True, 
                      kwargs_handlers=[InitProcessGroupKwargs(timeout=distributed_timeout)])
    with open(os.path.join(args.output_dir, 'run.log'), 'a+') as f:  
        f.write('\n' + '==' * 20 + '\n' + f"run at {datetime.now().strftime(r'%Y-%m-%d %H:%M:%S')}".center(40, '=') + '\n' + '==' * 20 + '\n')

    logging.info(f'distributed timeout: {distributed_timeout}')
    # print(torch.randint(1, 10, (1, 10)))
    if args.use_cached_data:  # data caching utils
        data_cache_dir = '/tmp/para_pretrain_data/'
        os.makedirs(data_cache_dir, exist_ok=True)
        train_data_file_name = os.path.split(args.dataset.data_file_path)[-1] # extract file name
        train_index_file_name = os.path.split(args.dataset.index_file_path)[-1]
        train_data_cache_path = os.path.join(data_cache_dir, train_data_file_name)
        train_index_cache_path = os.path.join(data_cache_dir, train_index_file_name)
        if acc.is_main_process:
            if not os.path.exists(train_data_cache_path) or args.force_recache:
                logging.info(f'caching data in {train_data_cache_path}')
                os.system(f'cp {args.dataset.data_file_path} {train_data_cache_path}')
            if not os.path.exists(train_index_cache_path) or args.force_recache:
                logging.info(f'caching index in {train_index_cache_path}')
                os.system(f'cp {args.dataset.index_file_path} {train_index_cache_path}')
        acc.wait_for_everyone()
        args.dataset.data_file_path = train_data_cache_path
        args.dataset.index_file_path = train_index_cache_path

    logging.info('initialize tokenizer, dataset and dataloader...')
    if not args.num_workers:
        args.num_workers = os.cpu_count() // acc.num_processes
        logging.info(f'num_workers not set, setting it to {args.num_workers} '
                     f'(num_cpu_cores: {os.cpu_count()}, num_distributed_processes: {acc.num_processes})')
    tok: T5Tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    if args.num_epochs == 1 and not args.finetune and args.resume and args.skip_data_in_dataset:
        skip_iterations = json.load(open(os.path.join(args.output_dir, 'checkpoints/training_state/run_state.json', ), 'r'))['current_iteration']
        skip_iterations *= args.batch_size
        logging.info(f'initiazling truncated dataset skipping first {skip_iterations} instances..')
        dataset = ParaPretrainMultitaskCurriculumDataset(**args.dataset, tokenizer=tok, start_idx=skip_iterations)
    else:
        dataset = ParaPretrainMultitaskCurriculumDataset(**args.dataset, tokenizer=tok)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_fn, prefetch_factor=args.prefetch_factor, pin_memory=True, pin_memory_device=acc.local_process_index, shuffle=True)
    val_dataset = ParaPretrainMultitaskValDataset(args.val_data_dir, tok, args.val_tasks)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, collate_fn=val_dataset.collate_fn, prefetch_factor=args.prefetch_factor)
    evaluator = ParaPretrainMultitaskEvaluator(**args.evaluator, tokenizer=tok)
        # initialize model 
    logging.info('initializing model...')
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    # original_embedding_size = model.get_input_embeddings().weight.data.shape(0)
    # expand model's embedding layer to fit tokenizer vocabulary size
    prev_embedding_dim = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    logging.info(f'model embedding size expanded from {prev_embedding_dim} to {model.get_input_embeddings().weight.shape[0]}')
    # start training
    train(acc, args, dataloader, val_dataloader, tok, model, evaluator)
    # parser.add_argument()

        
if __name__ == '__main__':
    main()
# height selection height: 3 tree: ( <node_45> ( <node_75> ( <node_69> ( <node_19> ( <node_44> <node_13> )) ( <node_5> ( <node_20> <node_68> ) ( <node_25> ( <node_50> ( <node_83> <node_39> ) ( <node_4> <node_24> )) ( <node_85> ( <node_93> ( <node_43> ( <node_12> <node_55> )) ( <node_10> ( <node_53> <node_52> ) ( <node_61> ( <node_57> <node_89> ) ( <node_87> <node_35> )))))))) ( <node_67> <node_73> ) ( <node_38> ( <node_66> <node_81> )) ( <node_32> <node_58> )))