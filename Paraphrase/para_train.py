from transformers import (
    T5Config, BartConfig, 
    T5Tokenizer, BartTokenizer,
    T5ForConditionalGeneration, BartForConditionalGeneration
)#from transformers.generation import beam_search
from torch.utils.bottleneck import __main__
import re
import torch
import sys
import os
import json
from typing import *
from transformers import T5ForConditionalGeneration
from torch import Tensor
from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch import distributed as dist
from torch.distributed import init_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from time import sleep, time
import pickle as pkl
import logging
import random as rd
import numpy as np
import subprocess
from copy import deepcopy
import re
from traceback import format_exc

# sys.path.append('..')
# try:
from .para_datasets import ParaNMTDataset, preprocess, BlockDistributedSampler
from .para_generating_utils import (
    teacher_forcing_generate, 
    val_test
)
# except ImportError:
#     sys.path.append('..')
#     from para_datasets import ParaNMTDataset, preprocess, BlockDistributedSampler
#     from para_generating_utils import (
#         teacher_forcing_generate, 
#         val_test
#     )
# try:
#     from para_train_data_utils import get_datasets_and_dataloaders
# except ImportError:
from .para_train_data_utils import get_datasets_and_dataloaders

from .post_process import (
    write_metrics_to_tensorboard,
    main as post_process_main
)

try:
    from jsonargparse import ArgumentParser, ActionConfigFile
    JSONARGPARSE_AVAILABLE = True
except ImportError as e:
    from argparse import ArgumentParser
    JSONARGPARSE_AVAILABLE = False

@record
def main():
    parser = ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--deepspeed', action='store_true', help='whether or not use deepspeed (train LLM)')
    parser.add_argument('-o', '--output-dir', help='directory to store checkpoints, validation results', type=str)
    # parser.add_argument('-s', '--serialization-path', type=str)  # 该项在训练的过程中为模型权重存储 文件夹 ，测试的过程中为模型权重文件
    parser.add_argument('-m', '--model', type=str, help='path or huggingface repo name of model', default='../pretrained-models/t5-base/')
    parser.add_argument('-T', '--tokenizer', type=str, help='path or huggingface repo name of tokenizer, default to `null`(corresponding to the same as `--model`)')
    parser.add_argument('--model-type', type=str, help='t5, bart, etc')
    parser.add_argument('-M', '--max-length', type=int, default=256, help='max number of tokens when loading data')
    parser.add_argument('--use-checkpoint', type=str, help='the checkpoint to use in the path specified by `--output-dir`')
    parser.add_argument('--num-warmup-epochs', type=int, default=1)
    parser.add_argument('--cosine-multiplier', type=float, default=1.2, 
        help="the lr_scheduler's total number of steps (steps for lr to decrease to 0) is multiplied by this")
    parser.add_argument('--target-only', action='store_true', 
        help='(only used for BART model) whether or not make BART model learn to only generate target sentence and parse tree')
    parser.add_argument('--use-tgt-as-ref', action='store_true', 
        help='use target syntax parse as exemplar syntax parse, for comparasion with target-based methods (GuiG, SI-SCP)')
    parser.add_argument('--use-num-postfix', action='store_true', help='whether or not the trees in ') 
    parser.add_argument('--instance-format-name', type=str, choices=['<xxx_tree>', 't5 prefix', '<tree>'], help='the way of splitting source sentence, source tree and exemplar/target tree ')
    parser.add_argument('-t', '--test-only', action='store_true', help='if test only be set to `True`, will only proceed inference on test set')
    parser.add_argument('-d', '--data', type=str, default='ParaNMT50m', 
        help='dataset name corresponding to the data directory (consisting of `train`, `test`, `val` subfolders, each consisting to src, (ref) tgt sentences and trees)')
    parser.add_argument('--use-opti', action='store_true', help='whether or not use trees stored in `[split].txt-corenlp-opti` (data processed by SGCP and AESOP)')
    parser.add_argument('--data-check', action='store_true', help='whether or not check data provided by dataloaders')
    parser.add_argument('--generation-debug', action='store_true', help='whether or not check generation outputs at real time')
    # parser.add_argument('--')
    parser.add_argument('--subset', action='store_true', help='whether or not use a subset of training set to proceed on training')
    parser.add_argument('--subset-cnt', type=float, help='use a subset of data (for debugging and testing), if value < 1, value fraction of total training set will be used, if value > 1, value instances from training data will be used')
    parser.add_argument('--logging-level', type=int, default=20, help='level of logging (defaut to 20 (logging.DEBUG))')
    parser.add_argument('-g', '--gradient-accumulation-steps', type=int, default=8)
    parser.add_argument('-W', '--num-workers', type=int, default=0)
    parser.add_argument('-F', '--prefetch-factor', type=int, default=2)
    parser.add_argument('-p', '--prune-height', type=int, default=4)
    parser.add_argument('--skip-post-process', action='store_true', 
        help='if set to `True`, metrics will not be calculated (`post process` means extracting generated target sentences from model outputs and calculate metrics)')
    parser.add_argument('-f', '--find-lr', action='store_true', help='whether or not proceed lr finding experiment')
    parser.add_argument('--decide-generation-kwargs', action='store_true', 
        help='whether or not do generation kwargs experiments, if set to `True`, `generation_kwargs` will reside on `generation_kwargs.json`')
    parser.add_argument('-H', '--half', action='store_true', 
        help='whether or not half the batch size and double gradient accumulation steps (for running on low VRAM machines)')
    if JSONARGPARSE_AVAILABLE:
        parser.add_argument('--config', action=ActionConfigFile)
    # print(f'set parser {type(parser)}')
    opt = parser.parse_args()
    # set seeds
    if 'SEED' in os.environ:
        seed = int(os.environ['SEED'])
    else:
        seed = 1453
    
    if opt.half:
        opt.batch_size = int(opt.batch_size / 2)
        opt.gradient_accumulation_steps = int(opt.gradient_accumulation_steps) * 2

    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # elastic multiprocessing (torchrun) issues
    # for code consistency, we use torchrun whatever gpu number we have
    rank = int(getattr(os.environ, 'RANK', 0))
    local_rank = int(getattr(os.environ, 'LOCAL_RANK', 0))
    world_size = int(getattr(os.environ, 'WORLD_SIZE', 1))
    # master_addr = os.environ['MASTER_ADDR']
    # master_port = os.environ['MASTER_PORT']
    ddp_ignored = (world_size == 1)
    if not ddp_ignored:
        init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
   
    # in mtai training clusters, docker names are ended with `pc[0-9]*`
    hostname_pc = re.search(r'pc\d+', subprocess.run(['hostname'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()).group(0)
    # a format string displaying time, level, absolute filepath: line number
    logging_fmt_str = f"%(asctime)s - {hostname_pc}:%(name)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s"
    # logging_fmt_str = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s"
    opt.gas = opt.gradient_accumulation_steps # `gas` is short for gradient accumulation steps
    if opt.model_type is None:
        opt.model_type = 't5' if 't5' in opt.model else 'bart' if 'bart' in opt.model else 'others'
    TYPE_TO_MODEL = {
        't5': (T5Config, T5Tokenizer, T5ForConditionalGeneration),
        'bart': (BartConfig, BartTokenizer, BartForConditionalGeneration)
    }

    # make output dir (checkpoints, results)
    if opt.output_dir is not None:
        checkpoint_dir = os.path.join(opt.output_dir, 'checkpoints')
        results_dir_exemplar = os.path.join(opt.output_dir, 'results', 'exemplar')
        results_dir_target = os.path.join(opt.output_dir, 'results', 'target')
        # if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir_exemplar, exist_ok=True)
        if 'triplet' not in opt.instance_format_name:
            os.makedirs(results_dir_target, exist_ok=True)
        # save config file if parser is jsonargparse.ArgumentParser
        if JSONARGPARSE_AVAILABLE:
            opt_without_gas = deepcopy(opt)
            opt_without_gas.pop('gas')
            parser.save(opt_without_gas, os.path.join(opt.output_dir, 'config.yaml'), overwrite=True)
        # initialize logging console and file handlers
        handlers=[
            logging.StreamHandler(), 
        ]
        if not opt.decide_generation_kwargs:
            handlers.append(logging.FileHandler(os.path.join(opt.output_dir, 'run.log')))

        if local_rank == 0 and not opt.test_only: 
            if not opt.use_checkpoint: # remove tensorboard log files if not resuming from training
                os.system(f'rm {opt.output_dir}/events.out.tfevents*')
            sw = SummaryWriter(log_dir=opt.output_dir, flush_secs=30)
        logging.basicConfig(format=logging_fmt_str, level=opt.logging_level, handlers=handlers)
        logger = logging.getLogger(f"para_train_r{local_rank}")
        logger.info(f'outputting to `{opt.output_dir}...`')

    # make serialization dir (1.pth, 2.pth, 3.pth, etc)
    # if opt.serialization_path is not None:
    #     if not os.path.exists(opt.serialization_path):
    #         os.mkdir(opt.serialization_path)

    else:
        logging.basicConfig(format=logging_fmt_str, level=opt.logging_level)
        logger = logging.getLogger("para_train (silent)")
        logger.info('silent(no output for validation)')

    # logger.info(f'Initializing ')
    # config: PretrainedConfig = TYPE_TO_MODEL[opt.model_type][0].from_pretrained(opt.model)
    # tokenizer: PreTrainedTokenizer = TYPE_TO_MODEL[opt.model_type][1].from_pretrained(opt.model)
    # model: PreTrainedModel = TYPE_TO_MODEL[opt.model_type][2](config).to(device)

    tokenizer_kwargs = {
        'return_tensors': 'pt',
        'truncation': True,
        'max_length': 200,
        'padding': True,
    }
    logger.info(f'Initializing model config, tokenizer and model (from {opt.model})...')
    checkpoint_dir = os.path.join(opt.output_dir, 'checkpoints')
    available_checkpoints = os.listdir(checkpoint_dir)
    if available_checkpoints and opt.use_checkpoint:
        # print(opt.use_checkpoint)
        latest_checkpoint = str(sorted(list(map(int, available_checkpoints)))[-1])
        if opt.use_checkpoint == 'latest':
            used_checkpoint = latest_checkpoint
            logger.info(f'Loading tokenizer and model parameters from checkpoint `latest({used_checkpoint})`')
        else:
            if opt.use_checkpoint in available_checkpoints:
                used_checkpoint = opt.use_checkpoint
                logger.info(f'Loading tokenizer and model parameters from checkpoint specified folder({used_checkpoint})')
            else:
                logger.warning(f"can't find checkpoint `{opt.use_checkpoint}`, use `latest({latest_checkpoint})` instead")
                used_checkpoint = latest_checkpoint
        ckpt_path = os.path.join(opt.output_dir, 'checkpoints', used_checkpoint)
        model = TYPE_TO_MODEL[opt.model_type][2].from_pretrained(ckpt_path).to(local_rank)
        if not ddp_ignored:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        tokenizer: PreTrainedTokenizer = TYPE_TO_MODEL[opt.model_type][1].from_pretrained(ckpt_path)
    else:
        logger.info(f'Initialize from huggingface checkpoint `{opt.model}`')
        model = TYPE_TO_MODEL[opt.model_type][2].from_pretrained(opt.model).to(local_rank)
        if not ddp_ignored:
            model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        tokenizer: PreTrainedTokenizer = TYPE_TO_MODEL[opt.model_type][1].from_pretrained(opt.tokenizer if opt.tokenizer is not None else opt.model)
 

    # if opt.model_type == 'bart':
    # if opt.instance_format_name != 't5_prefix':
    # add special tokens according to `instance_format_name`
    additional_tokens_dict = {}
    if opt.instance_format_name == '<xxx_tree>':
        additional_tokens_dict = {'additional_special_tokens': ['<src>', '<src_t>', '<temp>', '<temp_t>']}
    elif opt.instance_format_name == '<tree>':
        additional_tokens_dict = {'additional_special_tokens': ['<sent>', '<tree>']}
    elif opt.instance_format_name in ['aesop',]:
        additional_tokens_dict = {'additional_special_tokens': ['<sep>']}
    elif opt.instance_format_name == 'triplet_token':
        additional_tokens_dict = {'additional_special_tokens': ['<src_tree>', '<temp_tree>', '<tgt>']}
    elif opt.instance_format_name in ['triplet_tgtonly', 'triplet_tree', 'triplet_sentencetree', 'triplet_commontree', 'triplet_sentencetgttree']:
        additional_tokens_dict = {'additional_special_tokens': ['<tgt>']}

    if additional_tokens_dict:
        added_token_cnt = tokenizer.add_special_tokens(additional_tokens_dict, replace_additional_special_tokens=False)
        logging.info(f'{added_token_cnt} special tokens added to tokenizer')

    tokenizer.NT_set = [*(each for each in pkl.load(open('./data/ParaNMT50m_original/NT.pkl', 'rb')) if each)]
    added_token_cnt = tokenizer.add_tokens(tokenizer.NT_set)
    logging.info(f'{added_token_cnt} syntax tokens added to tokenizer')
    if opt.model_type == 'bart':
        tokenizer.NT_idx_set = set(tokenizer.convert_tokens_to_ids([*tokenizer.NT_set]))
        tokenizer.NT_idx_set |= set([tokenizer(' ' + each, add_special_tokens=False).input_ids[0] for each in tokenizer.NT_set if each])
    if not ddp_ignored:
        model.module.resize_token_embeddings(len(tokenizer))
    else:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f'Initializing datasets and dataloaders...')

    train_loader, val_loader_exemplar, val_loader_target, test_loader_exemplar, test_loader_target = get_datasets_and_dataloaders(
        opt, tokenizer
    )
    # method of extracting results from generated sequences
    test_dataset = test_loader_exemplar.dataset
    if isinstance(test_dataset, Subset):
        test_dataset = test_dataset.dataset
    extract_method = test_dataset.instance_format_name2extract_method[opt.instance_format_name]
    constraints = test_dataset.instance_format2constraint[opt.instance_format_name]
    if not opt.test_only:
        num_train_iters = len(train_loader)
        # initialize optimizer
        if not opt.find_lr:
            optimizer_model = AdamW(model.parameters(), opt.lr)
            scheduler_model = get_cosine_schedule_with_warmup(optimizer_model,
                opt.num_warmup_epochs * (num_train_iters // opt.gas + 1),
                opt.num_epochs * opt.cosine_multiplier * (num_train_iters // opt.gas + 1))
            if opt.use_checkpoint:
                optimizer_serialization_path = os.path.join(ckpt_path, 'optimizer.pt')
                scheduler_serialization_path = os.path.join(ckpt_path, 'scheduler.pt')
                # resume training optimizer and scheduler
                if os.path.exists(optimizer_serialization_path): 
                    optimizer_model.load_state_dict(optimizer_serialization_path)
                if os.path.exists(scheduler_serialization_path): 
                    scheduler_model.load_state_dict(scheduler_serialization_path)
        else:
            optimizer_model = torch.optim.SGD(model.parameters(), opt.lr)
        
    # tree_nonleaf_set = {'QP', 'LST', 'POS', 'NNP', 'RRC', 'NAC', 'VBD', 'VP', ':', 'DT', 'CD', 'WHADVP', ',', 'JJ', 'VBP',
    #                     'VBZ', 'SQ', 'CC', 'NP', 'ROOT', 'WP', 'RBS', 'FRAG', 'FW', 'PRN', 'EX', 'TO', 'WP$', 'WDT', 'IN',
    #                     'CONJP', 'SBAR', 'PRT', "''", 'LS', '.', 'PP', 'WHADJP', 'PDT', 'MD', 'VBN', 'PRP', 'S', 'RB',
    #                     'ADJP', 'VBG', 'SYM', 'UH', 'WRB', 'NNS', 'X', 'PRP$', 'JJR', 'WHPP', 'NNPS', 'VB', 'NN', 'SBARQ',
    #                     'JJS', 'INTJ', 'RP', 'RRB', 'SINV', 'UCP', 'RBR', '``', 'LRB', 'ADVP', 'NX', 'WHNP'}

    # def syntax_tree_to_sentence(t: str):
    #     t = re.sub(r'[\(\)]', '', t)
    if opt.decide_generation_kwargs and os.path.exists('generation_kwargs.json'):
        logger.info(f'Since this is a generation kwargs test run, '\
                   f'we use generation kwargs serilized from `{os.path.join(os.getcwd(), "generation_kwargs.json")}`')
        generation_kwargs = json.load(open('generation_kwargs.json', 'r'))
    else: 
        if 'triplet' in opt.instance_format_name and not opt.subset:
            gen_max_length = 400
        elif opt.subset and opt.subset_cnt > 1: 
            # if `subset` is set and subset_cnt is set to an integer > 1, we assume that we are entering a `debug` mod
            logging.info(f'setting generation `max_length` to 40 due to debugging')
            gen_max_length = 40
        else:
            gen_max_length = 256
        generation_kwargs = {
        "min_length": 5,
        "max_length": gen_max_length,
        "temperature": 1.0,
        "do_sample": False,
        "top_k": 0,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,# if 'triplet' not in opt.instance_format_name else 2.0,
        "num_beams": 1 if opt.generation_debug else 5,
        "no_repeat_ngram_size": 3
        # "bad_words_ids": [[628], [198]]
    }
    print(generation_kwargs)

    if not opt.test_only:
        # total_iteration = 0
        if not opt.find_lr:
            epoch_range = range(1, opt.num_epochs + 1) if not opt.use_checkpoint else range(1 + int(used_checkpoint), opt.num_epochs + 1)
            for epoch in epoch_range:
                total_loss = 0.0
                if not ddp_ignored:
                    model.module.train()
                else:
                    model.train()
                if local_rank == 0:
                    print('==' * 40 + '\n', end='')
                    print(f'Epoch {epoch}'.center(80) + '\n', end='')
                    print('==' * 40 + '\n', end='')
                logger.info(f'training on epoch ({epoch}/{opt.num_epochs})...')
                # print('=' * 40)
                # print(f'Epoch {epoch}'.center(40))
                # print('=' * 40)
                sleep(0.05)
                if local_rank == 0:
                    tqdm_train_loader = tqdm(total=len(train_loader), mininterval=2., ncols=100)
                for data_idx, batch in enumerate(train_loader):
                    if opt.data_check:
                        filter_pad = lambda x: x.replace('<pad>', '')
                        print('inputs:')
                        print(*map(filter_pad, tokenizer.batch_decode(batch.input_ids)), sep='\n')
                        print('labels:')
                        with torch.no_grad():
                            converted_labels = batch.labels.clone()
                            converted_labels[converted_labels == -100] = tokenizer.pad_token_id
                        print(*map(filter_pad, tokenizer.batch_decode(converted_labels)), sep='\n')
                        breakpoint()
                    # print(inputs, labels)
                    # tokenized_inputs = tokenizer(inputs, text_target=labels, **tokenizer_kwargs).to(device)
                    if not ddp_ignored:
                        model_res = model(**batch)
                    else:
                        model_res = model(**batch.to(local_rank))
                    loss = model_res.loss
                    if local_rank == 0: sw.add_scalar(f'loss', loss, (epoch - 1) * len(train_loader) + data_idx)
                    total_loss += loss.item()
                    if local_rank == 0:
                        tqdm_train_loader.set_postfix({'loss': loss.item()})
                    loss = loss / opt.gas
                    loss.backward()
                    if data_idx % (opt.gas - 1) == 0 or data_idx == len(train_loader) - 1:
                        optimizer_model.step()
                        scheduler_model.step()
                        optimizer_model.zero_grad()
                    if local_rank == 0:
                        tqdm_train_loader.update()

                val_test(opt=opt, tokenizer=tokenizer, model=model.module if not ddp_ignored else model,
                    current_epoch=epoch, data_mode='exemplar', loader=val_loader_exemplar, generation_args=generation_kwargs, constraints=constraints)
                if 'triplet' not in opt.instance_format_name:
                    val_test(opt=opt, tokenizer=tokenizer, model=model.module if not ddp_ignored else model,
                        current_epoch=epoch, data_mode='target', loader=val_loader_target, generation_args=generation_kwargs, constraints=constraints)
                if local_rank == 0:
                    serialization_path = os.path.join(checkpoint_dir, f'{epoch}')
                    os.makedirs(serialization_path, exist_ok=True)
                    if not ddp_ignored:
                        model.module.save_pretrained(serialization_path)
                    else: 
                        model.save_pretrained(serialization_path)
                    tokenizer.save_pretrained(serialization_path)
                    torch.save(optimizer_model.state_dict(), os.path.join(serialization_path, 'optimizer.pt'))
                    torch.save(scheduler_model.state_dict(), os.path.join(serialization_path, 'scheduler.pt'))
                if not opt.skip_post_process:
                    # try:
                    metrics_df_exemplar = post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='exemplar', dataset=val_loader_exemplar.dataset.data_dir, extract_method=extract_method, epoch=epoch, num_chunks=1 if ddp_ignored else None)
                    # except AssertionError as e:
                    #     metrics_df_exemplar = format_exc()
                    #     logging.warning(f'raised assertion error at epoch {epoch} mode exemplar {metrics_df_exemplar}:')
                    #     if 'evaluation' in os.getcwd():
                    #         os.chdir('..')
                    write_metrics_to_tensorboard(metrics_df_exemplar, epoch, 'exemplar', sw)
                    if 'triplet' not in opt.instance_format_name: # if not triplet based data, also do `target-only` evaluation
                        # try:
                        metrics_df_target = post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='target', dataset=val_loader_target.dataset.data_dir, extract_method=extract_method, epoch=epoch, num_chunks=1 if ddp_ignored else None)
                        # except AssertionError as e:
                            # metrics_df_target = format_exc()
                            # logging.warning(f'raised assertion error at epoch {epoch} mode target {metrics_df_target}:')
                            # if 'evaluation' in os.getcwd():
                            #     os.chdir('..')
                        write_metrics_to_tensorboard(metrics_df_target, epoch, 'target', sw)

            # logger.info(f'Calculating Metrics...')
            # if not opt.skip_post_process:
            #     post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='exemplar', dataset=opt.data, extract_method=extract_method, um_chunks=1 if ddp_ignored else None)
            #     if 'triplet' not in opt.instance_format_name: # if not triplet based data, also do `target-only` evaluation
            #         post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='target', dataset=opt.data, extract_method=extract_method, um_chunks=1 if ddp_ignored else None)
        else:
            import math
            import matplotlib.pyplot as plt
            def set_lr(optimizer: torch.optim.Optimizer, lr: float):
                for each in optimizer.param_groups:
                    each['lr'] = lr

            initial_lr = 1e-5
            max_lr = 0.1
            cyclic_steps = int(len(train_loader) / 4)
            lr_scale_factor = (max_lr / initial_lr) ** (1 / cyclic_steps)
            beta = 0.9
            current_scaled_beta = 0.9  # each time we calculate smoothed loss, we divide the exponential average with (1 - beta ^ step), this is `beta ^ step`
            current_lr = initial_lr
            avrg_loss = 0
            total_lrs = []
            total_smoothed_avrg_losses = []
            best_loss = 1e9
            pbar = trange(0, cyclic_steps, 1)
            pbar.write(f'Attempting to find optimal lr (lb={initial_lr}, ub={max_lr}, steps={cyclic_steps}, scale={lr_scale_factor})')
            for data_idx, batch in enumerate(train_loader):
                # training 
                set_lr(optimizer_model, current_lr)
                model_res = model(**batch)
                loss = model_res.loss
                loss.backward()
                optimizer_model.step()
                optimizer_model.zero_grad()
                loss = loss.item()
                # recording lr and loss
                # calculating exponential averaged and smoothed lr
                avrg_loss = beta * avrg_loss + (1 - beta) * loss
                smoothed_avrg_loss = avrg_loss / (1 - current_scaled_beta)
                pbar.set_postfix({'loss': smoothed_avrg_loss, 'lr': current_lr})
                if smoothed_avrg_loss < best_loss:
                    best_loss = smoothed_avrg_loss
                if smoothed_avrg_loss > best_loss * 4 or data_idx > cyclic_steps:
                    break
                # record them
                total_smoothed_avrg_losses.append(smoothed_avrg_loss)
                total_lrs.append(math.log10(current_lr))
                # modify lr and beta ^ step
                current_scaled_beta *= beta
                current_lr *= lr_scale_factor
                pbar.update()
            
            plt.figure()
            plt.plot(total_lrs, total_smoothed_avrg_losses,)
            plt.savefig(os.path.join(opt.output_dir, 'loss_lr_fig.png'), dpi=400)
            pkl.dump({'losses': total_smoothed_avrg_losses, 'lrs': total_lrs}, open(os.path.join(opt.output_dir, 'find_lr_data.pkl'), 'wb'))
            

    else:
        logger.info(f'test only, running test on checkpoint {opt.use_checkpoint}...')
        val_test(opt=opt, tokenizer=tokenizer, model=model.module if not ddp_ignored else model,
            current_epoch='test', data_mode='exemplar', loader=test_loader_exemplar, generation_args=generation_kwargs, constraints=constraints)
        if 'triplet' not in opt.instance_format_name:
            val_test(opt=opt, tokenizer=tokenizer, model=model.module if not ddp_ignored else model,
                current_epoch='test', data_mode='target', loader=test_loader_target, generation_args=generation_kwargs, constraints=constraints)
        if not opt.skip_post_process:
            post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='exemplar', dataset=test_loader_exemplar.dataset.data_dir, epoch='test', extract_method=extract_method, num_chunks=1 if ddp_ignored else None)
            if 'triplet' not in opt.instance_format_name:
                post_process_main(re.sub(r'(\./)?runs/', '', opt.output_dir), data_mode='target', dataset=test_loader_target.dataset.data_dir, epoch='test', extract_method=extract_method, um_chunks=1 if ddp_ignored else None)


if __name__ == '__main__':
    main()
