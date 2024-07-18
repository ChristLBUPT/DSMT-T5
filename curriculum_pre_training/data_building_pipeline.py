import sys; sys.path.append('..')
from para_pretrain_multitask_dataset import ParaPretrainMultitaskDataset, ParaPretrainTreeMultitaskDataset
from curriculum_pre_training.tree_task_data_utils import TreeTaskFormatter, multiprocessing_process
# from para_nmt_curriculum_utils import ParaPretrainMultitaskSplitter
from transformers import T5Tokenizer
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
import random as rd
from numba import jit, njit
import numba
from typing import List, Union, Optional, Type, Callable, Literal
import h5py
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from Paraphrase.para_datasets import ParaNMTDataset
import logging
import os
import re
import json
import pickle as pkl
import itertools
from typing import Tuple


class ParaPretrainMultitaskSplitter:
    def __init__(self, dataset: Dataset, tokenizer, export_path: str, export_tokenizer: bool = False, tokenizer_export_path: str = None, NT_path: str = None, ):
        """
        dataset: Dataset instance which reads a hdf5 file consisting of trees and yields inputs and labels formalized by a series of tasks
        export_path: path to export labeled and splitted (by tasks) data
        export_tokenizer: whether or not add a series of node tokens and save the tokenizer
        tokenizer_path: path to the PreTrainedTokenzier used to tokenize the file
        NT_path: path to the pickle file consisting of terminal syntax nodes
        """
        self.tokenizer = tokenizer
        if export_tokenizer:
            NT_tokens = [each for each in pkl.load(open(NT_path, 'rb')) if each]
            self.tokenizer.add_tokens(NT_tokens + ["<sep>"])
            self.tokenizer.save_pretrained(tokenizer_export_path)
        self.dataset = dataset
        # self.dataset = Subset(self.dataset, range(1000))
        # self.dataset.collate_fn = self.dataset.dataset.collate_fn
        self.num_workers = os.cpu_count() * 2
        # self.num_workers = 0
        self.dataloader = DataLoader(self.dataset, batch_size=self.num_workers or 1, num_workers=self.num_workers, collate_fn=self.dataset.collate_fn)
        self.task2sample_dict = defaultdict(list)
        self.tasks = []
        self.export_path = export_path
        self.indexer = [] # a list of [`t` (task_idx), `j` (index_within_task)] tuples indication the i-th data is the j-th data of that t-th task
        self.colon_idx = self.tokenizer.convert_tokens_to_ids(':')
    
    def split_data(self, max_num_data: int = -1):
        """Iterate over training data and split them by tasks(prefixes)"""
        current_idx = 0
        for batch in tqdm(self.dataloader, 'splitting samples..', total=max_num_data if max_num_data > 0 else len(self.dataloader)):
            # print(f'batch {current_idx}'.center(80, '='))
            for sample in batch:
                inputs, targets = sample
                # if (max(max(inputs), max(targets))) >= (1 << 15):
                #     print(f'too large elements found in ')
                prefix = self.tokenizer.decode(inputs[:inputs.index(self.colon_idx)])
                # prefix = prefix_pattern.match(sample).group(1)
                if prefix not in self.tasks:
                    self.tasks.append(prefix)
                # if 'span' not in prefix:
                #     print('input:', self.tokenizer.decode(inputs), sep='\n')
                #     print('label:', self.tokenizer.decode(targets), sep='\n')
                self.indexer.append((self.tasks.index(prefix), len(self.task2sample_dict[prefix])))
                self.task2sample_dict[prefix].append([np.array(partition, dtype=np.uint16) for partition in sample])
            # print('=' * 80)
            
            current_idx += 1
            if max_num_data > 0 and current_idx >= max_num_data:
                break
        
        with h5py.File(self.export_path, 'w') as f:
            logging.info(f'exporting to `{self.export_path}`')
            f.create_dataset('index', data=self.indexer)
            f.create_dataset('tasks', data=self.tasks)
            vlen_int_array_dtype = h5py.vlen_dtype(np.uint16)
            for task_prefix in self.task2sample_dict:
                inputs, labels = zip(*self.task2sample_dict[task_prefix])

                # f.create_dataset(task_prefix, data=np.array(self.task2sample_dict[task_prefix], dtype=[('inputs', 'O'), ('targets', 'O')]))
                logging.info(f'creating dataset `{task_prefix}.inputs`...')
                f.create_dataset(f'{task_prefix}/inputs', data=inputs, dtype=vlen_int_array_dtype)
                # if 'span' in task_prefix:
                #     print(f'for {task_prefix = }')
                #     print('top 10 samples:')
                #     print(self.tokenizer.batch_decode(inputs[:10]))
                #     print('=' * 80)
                if np.asarray(labels).dtype == np.dtype('O'):
                    logging.info(f'creating dataset `{task_prefix}.labels`...')
                    f.create_dataset(f'{task_prefix}/labels', data=labels, dtype=vlen_int_array_dtype)
                else:
                    f.create_dataset(f'{task_prefix}/labels', data=labels)


## ==============================================================================================================================
##              step 1: split unsupervised data (train_data.h5) into multitask data (train_data_multitask.h5)
##                      according to 
## ==============================================================================================================================
def split_data_by_task(
        unsupervised_file_path: str = '../data/ParaNMT50m_original/train_data.h5',
        multitask_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5',
        pretrained_t5_tokenizer_path: Optional[str] = '../pretrained-models/t5-base', 
        export_tokenizer: bool = False,
        syntax_t5_tokenizer_path: Optional[str] = '../pretrained-models/syntax-t5-base',
        NT_pickle_path: Optional[str] = '../data/ParaNMT50m_original/NT.pkl'
    ):
    """
    split unsupervised data (like `train_data.h5`) 
    which only constitutes linearized tree strings (data from `trees` key) 
    into multitask data (like `train_data_multitask.h5`), 
    which constitues data from multiple tasks (prefixes) seperately stored in different data keys' `inputs` and `labels`, 
    like `pruned_tree_parse/inputs` `pruned_tree_parse/labels`, 
    This procedure also produces an index file, which constitutes of 2-int-element tuples indicating the task index and the sample index within that task for each sample.
    """
    tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_tokenizer_path)
    dataset = ParaPretrainMultitaskDataset(unsupervised_file_path, tokenizer, dont_collate=True)
    spl = ParaPretrainMultitaskSplitter(
        dataset=dataset, tokenizer=tokenizer, export_path=multitask_file_path, export_tokenizer=export_tokenizer,
        tokenizer_export_path=syntax_t5_tokenizer_path if export_tokenizer else None,
        NT_path=NT_pickle_path if export_tokenizer else None
    )
    spl.split_data()

def split_tree_data_by_task(
        trees_file_path: str = '../data/ParaNMT50m_original/random_trees_<node>_bracket.h5',
        export_path: str = None,
        # trees_multitask_file_path: str = '../data/ParaNMT50m_original/random_trees_<node>_bracket_multitask.h5',
        tokenizer_path: str = '../pretrained-models/syntax-t5-base-node/',
        linearizing_format: Literal['bracket', 'slash'] = 'bracket',
):
    logging.basicConfig(level=logging.INFO)
    logging.info(f'splitting tree data into tasks ({trees_file_path =}, {linearizing_format =})')
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    trees_multitask_file_path = export_path or trees_file_path.replace(f'_{linearizing_format}', f'_{linearizing_format}_multitask')
    dataset = ParaPretrainTreeMultitaskDataset(trees_file_path, linearizing_format, tokenizer, dont_collate=True)
    spl = ParaPretrainMultitaskSplitter(dataset, tokenizer, trees_multitask_file_path)
    spl.split_data()


def get_tree_data_validation_set(
    tree_multitask_file: str = '../data/ParaNMT50m_original/random_trees_<node>_bracket_multitask.h5',
    tokenizer_path: str = '../pretrained-models/syntax-t5-base-node/',
    n_validation_samples_per_task: int = 1000,
    export_dir: str = '../data/ParaNMT50m_original/val_data_bracket_tree'
):
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    with h5py.File(tree_multitask_file, 'a') as f:
        tasks = f['tasks'][...]
        task_official_names = TreeTaskFormatter.tasks
        if isinstance(tasks[0], bytes): tasks = [*map(bytes.decode, tasks)]
        index = f['index'][...]
        num_samples = [f[task]['inputs'].shape[-1] for task in tasks]
        task_idx2val_min_index = {}
        for task_idx, task in enumerate(tasks):
            print(f'[{task_idx}] {task}'.center(80, '='))
            print(f'getting validation samples for task `{task}`')
            print(f'total number of samples: {num_samples[task_idx]}')
            val_min_index = num_samples[task_idx] - n_validation_samples_per_task # if index > `val_min_index`, this sample belongs to validation set
            task_idx2val_min_index[task_idx] = val_min_index
            val_sample_index_range = [*range(val_min_index, num_samples[task_idx])]
            instances = []
            for input_seq, target_seq in zip(tqdm(f[task]['inputs'][val_sample_index_range]), f[task]['labels'][val_sample_index_range]):
                input_seq = tokenizer.decode(input_seq).replace('</s>', '').strip()
                target_seq = tokenizer.decode(target_seq).replace('</s>', '').strip()
                instances.append({"input": input_seq, "target": target_seq, "task_idx": task_idx})
            
            with open(os.path.join(export_dir, f'{task_idx:02d}_{task_official_names[task_idx]}.json'), 'w') as f_out:
                json.dump(instances, f_out, indent=2)
        
        # new_index = [None for _ in range(sum(num_samples) - n_validation_samples_per_task * len(tasks))]
        # cptr = 0
        # for each in tqdm(index, desc='stripping out indices of validation samples'):
        #     if each[1] < task_idx2val_min_index[each[0]]:
        #         new_index[cptr] = each
        #         cptr += 1
        
        # print(f'writing new index to file...')
        # f['index'][...] = new_index

## ==============================================================================================================================
##      step 1.5: convert txt format scpg (`ParaNMT-small`) data file (src.txt, tgt.txt, ref.txt) to pre-training h5(train) and json(val) file
## ==============================================================================================================================

def serialize_scpg_data_h5(syntax_t5_tokenizer_path: str, scpg_data_root_dir: str, original_multitask_file_path: str, instance_format_name: str = 't5_prefix'):
    """fuse scpg data into the multitask data file specifed by `original_multitask_file_path`"""
    logging.basicConfig(format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d - %(message)s', level=logging.INFO)
    # initialize tokenizer, dataset and dataloader
    tokenizer = T5Tokenizer.from_pretrained(syntax_t5_tokenizer_path)
    d = ParaNMTDataset(os.path.join(scpg_data_root_dir, 'train'), 't5', tokenizer, 5, False, instance_format_name=instance_format_name)
    scpg_inputs, scpg_targets = [], []
    tqdm_dl = tqdm(DataLoader(d, batch_size=1, num_workers=os.cpu_count(), collate_fn=d.collate_fn), mininterval=1)
    tqdm_dl.set_description('iterating over data...')
    # iterate over data, fetch all inputs and targets
    for batch_idx, batch in enumerate(tqdm_dl):
        scpg_inputs.append(batch.input_ids[0].tolist())
        scpg_targets.append(batch.labels[0].tolist())

    for idx in range(len(scpg_inputs)):
        scpg_inputs[idx] = np.array(scpg_inputs[idx])
        scpg_targets[idx] = np.array(scpg_targets[idx])

    vlen_dtype = h5py.vlen_dtype(np.int32)
    with h5py.File(original_multitask_file_path, 'a') as f:
        f.create_dataset('scpg/inputs', data=scpg_inputs, dtype=vlen_dtype)
        f.create_dataset('scpg/labels', data=scpg_targets, dtype=vlen_dtype)

    with h5py.File(original_multitask_file_path, 'r') as f:
        for i in range(4):
            print(tokenizer.convert_ids_to_tokens(f['scpg']['inputs'][i]))
            print(tokenizer.convert_ids_to_tokens(f['scpg']['labels'][i]))
        print(f['scpg']['inputs'].size)

def serialize_scpg_data_json(syntax_t5_tokenizer_path: str, scpg_data_root_dir: str, export_file_path: str, instance_format_name: str = 't5_prefix'):
    tokenizer = T5Tokenizer.from_pretrained(syntax_t5_tokenizer_path)
    d = ParaNMTDataset(os.path.join(scpg_data_root_dir, 'val'), 't5', tokenizer, 5, True, instance_format_name=instance_format_name)
    import json
    t5_prefix_fmt = d.instance_formats[instance_format_name]
    with open(export_file_path, 'w') as f:
        instances = []
        for each in d.instances:
            instances.append({"input": t5_prefix_fmt.format(**each), "target": each['tgt_text'], "task_idx": 7})
        
        json.dump(instances, f, indent=2)

def serialize_scpg_data(
        syntax_t5_tokenizer_path: str, scpg_data_root_dir: str ='../data/ParaNMT50m/',
        export_file_dir: str = '../data/ParaNMT50m_original', original_multitask_file_path: str = 'train_data_multitask.h5', instance_format_name: str = 't5_prefix'):
    original_multitask_file_path = os.path.join(export_file_dir, original_multitask_file_path)
    available_indices = set(  # get all non-scpg task indices and append scpg to them
        map(
            int, 
            [re.match('^(\d+)_[a-z_]*\.[a-z]*', each).group(1) for each in filter(lambda name: 'scpg' not in name, os.listdir(os.path.join(export_file_dir, 'val_data')))]
        )
    )
    val_export_file_path = os.path.join(export_file_dir, 'val_data', f'{max(available_indices) + 1:02d}_scpg.json')
    print(f'validation data export path: `{val_export_file_path}`')
    serialize_scpg_data_h5(
        syntax_t5_tokenizer_path=syntax_t5_tokenizer_path,
        scpg_data_root_dir=scpg_data_root_dir,
        original_multitask_file_path=original_multitask_file_path,
        instance_format_name=instance_format_name
    )
    serialize_scpg_data_json(
        syntax_t5_tokenizer_path=syntax_t5_tokenizer_path,
        scpg_data_root_dir=scpg_data_root_dir,
        export_file_path=val_export_file_path,
        instance_format_name=instance_format_name
    )


def calculate_sample_length(index_term: Tuple[int, int], ):
    global f_multitask, f_index, task_names
    task_idx, sample_idx = index_term
    return f_multitask[task_names[task_idx]]['inputs'][sample_idx].shape[-1], f_multitask[task_names[task_idx]]['labels'][sample_idx].shape[-1]

def group_data_by_length(multitask_file_path: str, index_file_path: str, mini_batch_size: int = 32, n_shuffles: int = 5, read_from_cache: bool = False):
    global f_multitask, f_index, task_names
    with h5py.File(multitask_file_path, 'r') as f_multitask, h5py.File(index_file_path, 'r') as f_index:
        total_n_samples = len(f_index['index'])
        print(f'calculating lengths for file `{multitask_file_path}`...')
        task_names = [*map(bytes.decode, f_multitask['tasks'])]
        # for total_idx, (task_idx, sample_idx) in enumerate(tqdm(f_index['index'])):
        #     task_name = task_names[task_idx]
        #     lengths_inputs[total_idx] = f_multitask[task_name]['inputs'][sample_idx].shape[-1]
        #     lengths_labels[total_idx] = f_multitask[task_name]['labels'][sample_idx].shape[-1]
        if not read_from_cache:
            lengths = multiprocessing_process(f_index['index'], calculate_sample_length)
            pkl.dump(lengths, open('../pickles/data_building_pipeline/lengths.pkl', 'wb'))
        else:
            print(f'used cached file `../pickles/data_building_pipeline/lengths.pkl`')
            lengths = pkl.load(open('../pickles/data_building_pipeline/lengths.pkl', 'rb'))
        lengths_inputs, lengths_labels = zip(*lengths)
        
        idx2length = [*enumerate(lengths_inputs)]
        print(f'sorting by length descending...')
        idx2length.sort(key=lambda x: x[1], reverse=True)
        idx2length_chunks = [idx2length[chunk_start: chunk_start + mini_batch_size] for chunk_start in range(0, total_n_samples, mini_batch_size)]

        old_index = f_index['index'][:]
        index_file_dir, index_filename = os.path.split(index_file_path)
        for file_idx in range(n_shuffles):
            middle_chunks = idx2length_chunks[1: -1]
            rd.shuffle(middle_chunks)
            idx2length_chunks = [idx2length_chunks[0]] + middle_chunks + [idx2length_chunks[-1]]
            shuffled_index = [old_index[idx] for idx, _ in itertools.chain(*idx2length_chunks)]
            with h5py.File(os.path.join(index_file_dir, index_filename.replace('.h5', f'_grouped_reshuffled_{file_idx:02d}.h5')), 'w') as f:
                f.create_dataset('index', data=shuffled_index)
                f.create_dataset('tasks', data=f_index['tasks'][:])
                f.create_dataset('totals', data=[f_multitask[each]['inputs'].shape[-1] for each in task_names])




## ==============================================================================================================================
##              step 3: sample multitask data in a curriculum manner
## ==============================================================================================================================

## ==============================================================================================================================
##              you can implement this by calling `simulation_main` from `simulation_utils`
## ==============================================================================================================================


## ==============================================================================================================================
##              step 4 (optional): truncating data according to curriculum index file 
##                  (this is necessary if sample-based curriculum learning is used)
## ==============================================================================================================================

def calc_max_counts(f_index_in: h5py.File):
    tasks = [*map(bytes.decode, f_index_in['tasks'])]
    task_max_count = [0] * len(tasks)
    for task_idx, sample_idx in tqdm(f_index_in['index'], desc='calculating task max lengths'):
        if sample_idx > task_max_count[task_idx]:
            task_max_count[task_idx] = sample_idx
    
    return task_max_count


def truncate_data_by_index(data_file_path: str, index_file_path: str, export_file_path: str):
    with h5py.File(data_file_path, 'r') as f_data_in, h5py.File(index_file_path, 'r') as f_index_in, h5py.File(export_file_path, 'w') as f_data_out:
        # task_keys = [*filter(lambda x: x not in ['index', 'tasks'], f_data_in.keys())]
        tasks = [*map(bytes.decode, f_index_in['tasks'])]
        task_max_count = calc_max_counts(f_index_in)
        for task_name in tasks:
            print(f'exporting data for task `{task_name}`...')
            task_trunc_cnt = task_max_count[tasks.index(task_name)]
            f_data_out.create_dataset(f'{task_name}/inputs', data=f_data_in[task_name]['inputs'][:task_trunc_cnt])
            f_data_out.create_dataset(f'{task_name}/labels', data=f_data_in[task_name]['labels'][:task_trunc_cnt])

def create_truncated_index(data_file_path: str, export_index_file_path: str, proportion: float = 0.15):
    with h5py.File(data_file_path, 'r') as f_data_in, h5py.File(export_index_file_path, 'w') as f_index_out:
        totals = [int(len(f_data_in[each]['inputs']) * proportion) for each in f_data_in['tasks']]
        f_index_out.create_dataset('totals', data=totals)
        new_index = [[0, 0] for _ in range(sum(totals))]
        total_idx = 0
        for task_idx, data_idx in tqdm(f_data_in['index'], total=sum(totals)):
            if data_idx < totals[task_idx]:
                new_index[total_idx] = task_idx= [task_idx, data_idx]
                total_idx += 1
            if total_idx == sum(totals):
                break

        f_index_out.create_dataset('index', data=new_index)
        f_index_out.create_dataset('tasks', data=f_data_in['tasks'][:])



def strip_out_tasks_in_index_file(data_file_path: str, index_file_path: str, export_file_path: str, task_mask: List[int]):
    """strip out data for specific (masked) tasks and replace them with data of unmasked tasks"""
    with h5py.File(data_file_path, 'r') as f_in_data, h5py.File(index_file_path, 'r') as f_in_index, h5py.File(export_file_path, 'w') as f_out_index:
        # calculate total number of each tasks' data of input data file and input index file
        max_totals = []
        for task_name in map(bytes.decode, f_in_index['tasks']):
            max_totals.append(f_in_data[task_name]['inputs'].shape[-1])
        
        current_totals = f_in_index['totals'][:].tolist() if 'totals' in f_in_index.keys() else calc_max_counts(f_in_index)
        current_num_samples = sum(current_totals) # total number of samples
        unmasked_tasks = [*filter(lambda x: task_mask[x], range(len(task_mask)))]
        # filter out samples from masked tasks
        # new_index = [(0, 0) for _ in range(len(f_in_index['index']))]
        new_index = []
        for task_idx, sample_idx in tqdm(f_in_index['index'], desc='filtering out samples from masked tasks'):
            if task_mask[task_idx] == 1:
                new_index.append((task_idx, sample_idx))

        # sample new task indices
        for idx in trange(current_num_samples - len(new_index), desc='sampling new samples evenly'):
            unmasked_tasks_remainings = [max_totals[idx] - current_totals[idx] for idx in unmasked_tasks]
            sampled_task_idx = rd.choices(unmasked_tasks, unmasked_tasks_remainings)
            new_index.append(sampled_task_idx, current_totals[sampled_task_idx])
            current_totals[sampled_task_idx] += 1

        f_out_index.create_dataset('tasks', data=f_in_index['tasks'])
        f_out_index.create_dataset('index', data=new_index)
        # total number of data (set masked tasks to `0`)
        new_totals = [0] * range(len(current_totals))
        for idx, each in enumerate(current_totals):
            if task_mask[idx]:
                new_totals[idx] = each

        f_out_index.create_dataset('totals', data=new_totals)


if __name__ == "__main__":
    # multitask_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5',
    # pretrained_t5_tokenizer_path: Optional[str] = '../pretrained-models/t5-base', 
    # export_tokenizer: bool = False,
    # syntax_t5_tokenizer_path: Optional[str] = '../pretrained-models/syntax-t5-base',
    # NT_pickle_path: Optional[str] = '../data/ParaNMT50m_original/NT.pkl'
    # split_data_by_task(
    #     multitask_file_path='../data/ParaNMT50m_original/after_tree_train_data_multitask.h5', 
    #     pretrained_t5_tokenizer_path='../pretrained-models/syntax-t5-base-node-with-NT', export_tokenizer=False)
    # split_tree_data_by_task('../data/ParaNMT50m_original/train_data.h5', '../data/ParaNMT50m_original/trees_bracket_multitask.h5')
    # split_tree_data_by_task(linearizing_format='slash')
    # get_tree_data_validation_set()
    group_data_by_length(
        '../data/ParaNMT50m_original/after_tree_train_data_multitask.h5', 
        '../data/ParaNMT50m_original/after_tree_train_data_multitask_large_index.h5', read_from_cache=False, n_shuffles=1)
    # serialize_scpg_data(
    #     syntax_t5_tokenizer_path="../pretrained-models/syntax-t5-base",
    #     scpg_data_root_dir="../data/ParaNMT50m_triple",
    #     original_multitask_file_path='train_data_multitask_triplet_scpg.h5',
    #     instance_format_name='triplet_nooutputtree')
    # serialize_scpg_data_h5(
    #     '../pretrained-models/syntax-t5-base-node-with-NT/',
    #     '../data/ParaNMT50m/', '../data/ParaNMT50m_original/_after_tree_train_data_multitask.h5'
    # )    
    # create_truncated_index('../data/ParaNMT50m_original/after_tree_train_data_multitask.h5', '../data/ParaNMT50m_original/after_tree_train_data_multitask_large_index.h5', 0.8)
    # truncate_data_by_index(
    #     data_file_path='../data/ParaNMT50m_original/train_data_multitask.h5', 
    #     index_file_path='../data/ParaNMT50m_original/train_data_multitask_medium_with_scpg_index_curriculum.h5',
    #     export_file_path='../data/ParaNMT50m_original/train_data_multitask_medium_with_scpg_data.h5')
    # strip_out_tasks_in_index_file(
    #     '../data/ParaNMT50m_original/train_data_multitask.h5',
    #     '../data/ParaNMT50m_original/train_data_multitask_medium_with_full_scpg_index_random.h5',
    #     '../data/ParaNMT50m_original/train_data_multitask_medium_with_full_scpg_index_random_w_o_unsupervised.h5',
    #     [0,0,0,0,1,1,1,1,1,1,1,1]
    #     )
