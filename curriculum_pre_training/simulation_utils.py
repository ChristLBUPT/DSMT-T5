import sys; sys.path.append('..')
from tqdm import trange
from matplotlib import pyplot as plt
import random as rd
from numba import njit
from typing import List, Union
import h5py
import numpy as np
import os
import pathlib
import json

def read_multitask_data(
    multitask_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5', 
    # scpg_file_path: str = '../data/ParaNMT50m_original/train_data_scpg.h5',
    curriculum_difficulty_file_path: str = './config/data_utils/curriculum_difficulties.json',
    curriculum_difficulty_setting_name: str = 'default',
    category_mask: List[int] = None,
    multitask_data_cnt_divisor: float = 30,
    scpg_data_cnt_divisor: float = 10
    ):
    with open(curriculum_difficulty_file_path, 'r') as f:
        # load difficulty settings (the task at the latter position of the task list is harder)
        curriculum_difficulty_dict = json.load(f)
        category_names = curriculum_difficulty_dict[curriculum_difficulty_setting_name]
        if category_mask: # only select unmasked (value `1`) categories
            category_names = [each[0] for each in zip(category_names, category_mask) if each[1] == 1]
    with h5py.File(multitask_file_path, 'r') as f_multitask:
        # h5py.File(scpg_file_path, 'r') as f_scpg:
        print(f_multitask['index'][:20])
        # trunc multitask data according to `multitask_data_proportion`
        totals = np.array(
            [len(f_multitask[category_name]['inputs']) for category_name in filter(lambda x: 'scpg' not in x, category_names)]
            ) // multitask_data_cnt_divisor # only `1 / multitask_data_proportion` of total unsupervised data used for unsupervised training
        # add truncted scpg data count to totals
        if any(map(lambda x: 'scpg' in x, category_names)):
            for scpg_name in filter(lambda x: 'scpg' in x, category_names):
                totals = np.append(totals, len(f_multitask[scpg_name]['inputs']) // scpg_data_cnt_divisor)
        assert len(totals) == len(category_names), f"error, totals ({len(totals)}) != length of category_names ({len(category_names)})"
        return totals, category_names 
    
def get_scale_sequences_cosine(totals: np.ndarray, beta: float = 1.5):
    """
    Args:
        beta: starts will be [beta ^ N, beta ^ (N  - 1), ... beta] (N is the number of categories)
    returns [num_categories, num_iterations] every categories' scale (difficulty factor) during the whole iteration
    """
    num_categories = len(totals)
    categories = np.array([*range(num_categories)])
    starts = np.array([*(beta ** i for i in range(1, num_categories + 1))][::-1]) # 
    ends = np.array([*starts]) 

    num_iterations = sum(totals) # num_iterations (total number of samples)
    scale_sequences = np.zeros((num_categories, num_iterations))
    for category_idx in range(num_categories):
        x = np.linspace(0, np.pi, num_iterations)
        scale_sequences[category_idx] = (1 + np.cos(x)) / 2 * (starts[category_idx] - ends[category_idx]) \
            + ends[category_idx] # descend (or ascend) from `start` to `end` in cosine manner
    
    return scale_sequences

# def plot_scale_sequences(scale_sequences: np.ndarray, category_names: List[str], savefig_dir: str = './visualization', savefig_filename: str = 'cumsum.png'):
#     plt.figure()
#     plt.title('scale sequences')
#     num_categories, num_iterations = scale_sequences.shape
#     for category_idx in range(num_categories):
#         x = np.linspace(0, np.pi, num_iterations)
#         print(f'plotting {category_names[category_idx]} ({category_idx + 1}/{num_categories})')
#         plt.plot(x, scale_sequences[category_idx], label=category_names[category_idx])
#     plt.legend(loc='upper right')
#     plt.show()
#     print(scale_sequences[:,:100])

@njit
def sample_step(probability_cumsum: np.ndarray):
    """ samples a number (task index) according to a cumsum of probabilities [p1, p1 + p2, p1 + p2 + p3, ...]
    Args:
        probability_cumsum: cumsum of probabilities of all categories
    Returns: sampled number (task index)"""
    rand_number = np.random.random()
    intercepted_probability_cumsum = probability_cumsum - rand_number
    for idx in range(intercepted_probability_cumsum.shape[-1]):
        if intercepted_probability_cumsum[idx] >= 0:
            break
    return idx

@njit
def simulate_step(i: int, samples: np.ndarray, remainings: np.ndarray, scale_sequences: np.ndarray, percentage_sequences: np.ndarray):
    """simulation at step `i` (calculate current sampling rate/probability, cumsum them and sample a task idx according to them)"""
    this_probabilities = remainings * scale_sequences[:,i]
    this_probabilities = this_probabilities / this_probabilities.sum()
    probability_cumsum = this_probabilities.cumsum()
    sample = sample_step(probability_cumsum)
    if sample == -1:
        return -1
        
    samples[i] = sample
    remainings[int(samples[i])] -= 1
    percentage_sequences[:,i] = this_probabilities

    return 0

def simulate_numpy(
    samples: Union[List[int], np.ndarray],
    remainings: Union[List[int], np.ndarray],
    scale_sequences: Union[List[List[int]], np.ndarray], # [num_iterations, num_categories]
):
    """simulate from begin to end (call `simulate_step` for `num_iteration` steps)"""
    _, num_iterations = scale_sequences.shape
    percentage_sequences = np.zeros_like(scale_sequences)
    for i in trange(num_iterations):
        if simulate_step(i, samples=samples, remainings=remainings, scale_sequences=scale_sequences, percentage_sequences=percentage_sequences) == -1:
            break

    return samples, percentage_sequences

def mix_sample_sequences(sequences: List[np.ndarray], percentages: List[np.ndarray], probabilities: List[float], indexing_sequence: np.ndarray):
    """mix two or more sequences together according to mixing probabilities,
    if sequences has two elements, then `indexing_sequence` should be [0, 1, 0, 1, ...] (0 for first sequence, 1 for second sequence)"""
    sequence_lengths = [each.shape[-1] for each in sequences]
    total_num_iterations = sum(sequence_lengths)
    # sample samples from each sequence according to `indexing_sequence`
    samples = np.zeros(total_num_iterations, dtype=np.int32)
    percentages = np.zeros(total_num_iterations)
    pivots = [0] * len(sequences)
    for i in trange(total_num_iterations, desc='fusing examples...'):
        samples[i] = sequences[pivots[indexing_sequence[i]]]
        pivots[indexing_sequence[i]] += 1
    
    return samples

def plot_sample_cumsums(samples_arr, category_names: List[str], savefig_dir: str = './visualization', savefig_filename: str = 'cumsum.png'):
    category_cumsums = [np.cumsum(samples_arr == each) for each in range(len(category_names))] # [category 1 cumsum, category 2 cumsum, ...]

    plt.figure()
    x = np.arange(len(samples_arr))
    for category_idx, category_name in enumerate(category_names):
        print(f'plotting cumsum `{category_name}` ({category_idx + 1}/{len(category_names)})...')
        plt.plot(x, category_cumsums[category_idx], label=f'cs_{category_name}', linestyle='-' if category_name.find('span') > 0 else ':')
    
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(savefig_dir, savefig_filename))
    plt.show()

def plot_sample_scales(scale_sequences: np.ndarray, category_names: List[str], savefig_dir: str = './visualization', savefig_filename: str = 'scales.png'):
    """this is suitable for plotting scales and percentages (remainins * scales)"""
    plt.figure()
    x = np.arange(scale_sequences.shape[-1])
    legend_label_name = pathlib.Path(savefig_filename).stem.split('_')[0]
    for category_idx, category_name in enumerate(category_names):
        print(f'plotting scale `{category_name}` ({category_idx + 1}/{len(category_names)})...')
        plt.plot(x[:-10], scale_sequences[category_idx][:-10], label=f'{legend_label_name}_{category_name}', linestyle='-' if category_name.find('span') > 0 else ':')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(savefig_dir, savefig_filename))
    plt.show()

def serialize_curriculum_data(
    totals: np.ndarray,
    samples: np.ndarray,
    category_names: List[str],
    multitask_h5_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5',
    # scpg_h5_file_path: str = '../data/ParaNMT50m_original/train_data_scpg.h5',
    export_directory: str = '../data/ParaNMT50m_original',
    export_name: str = 'train_data_multitask_small_with_scpg', 
    export_data: bool = False,
    ):
        # h5py.File(scpg_h5_file_path, 'r') as f_scpg, \
    with h5py.File(multitask_h5_file_path, 'r') as f_in, \
        h5py.File(os.path.join(export_directory, f'{export_name}_index_random.h5'), 'w') as f_out_index_random, \
        h5py.File(os.path.join(export_directory, f'{export_name}_index_curriculum.h5'), 'w') as f_out_index_curriculum:
        vlen_int_array_dtype = h5py.vlen_dtype(np.uint16)
        # =========================(optional) form additional `data` h5py file=========================
        if export_data:
            with h5py.File(os.path.join(export_directory, f'{export_name}_data.h5'), 'w') as f_out_data:
                for key in category_names: # add 11 unsupervised task data to output `data` dataset
                    data_trunc_cnt = totals[category_names.index(key)] # if we use only a proportion of task `key`, this will make sense
                    print(f"creating dataset for {key}...(total {data_trunc_cnt} instances)")
                    f_out_data.create_dataset(f'{key}/inputs', data=f_in[key]['inputs'][:data_trunc_cnt], dtype=vlen_int_array_dtype)
                    f_out_data.create_dataset(f'{key}/labels', data=f_in[key]['labels'][:data_trunc_cnt], dtype=vlen_int_array_dtype)
        # add scpg data to output `data` dataset
        # scpg_data_cnt = totals[-1] # last in `totals` in scpg
        # f_out_data.create_dataset(f'scpg/inputs', data=f_scpg['scpg']['inputs'][:scpg_data_cnt], dtype=vlen_int_array_dtype)
        # f_out_data.create_dataset(f'scpg/labels', data=f_scpg['scpg']['labels'][:scpg_data_cnt], dtype=vlen_int_array_dtype)
        # f_out_index_random.create_dataset('tasks', data=np.append(f_in['tasks'], b'scpg'))
        # =========================(deprecated) form additional `data` h5py file=========================
        ## we need to inject scpg data into random index, it's mainly becuase of two reasone
        # 1. Original multitask data (train_data_multitask.h5) consists scpg data,
        #    but do not consist of scpg data indices 
        #    (it's `index` field formed by `ParaPretrainMultitaskSplitter.split_data()` only consists of these non-scpg unsupervised/supervised tasks)
        # 2. scpg data and other multitask data use different truncation counts,
        #    hence it's not possible to pre-compute a set of random indices corresponding to the proportions of them
        print('injiecting scpg data into random index...')
        scpg_data_cnt = totals[category_names.index('scpg')]
        num_categories = len(category_names)
        num_iterations = samples.shape[-1]
        scpg_probability = scpg_data_cnt / num_iterations
        current_unsupervised_idx = 0
        current_scpg_idx = 0
        mixed_index = [[0, 0] for _ in range(num_iterations)]
        trange_iteration = trange(num_iterations, mininterval=1)
        for idx in trange_iteration:
            if rd.random() < scpg_probability:
                if current_scpg_idx < scpg_data_cnt:
                    selection = 'scpg'
                else:
                    selection = 'multitask'
            else:
                if current_unsupervised_idx < num_iterations - scpg_data_cnt:
                    selection = 'multitask'
                else:
                    selection = 'scpg'
            if selection == 'scpg': # use scpg data
                mixed_index[idx] = [len(totals) - 1, current_scpg_idx]
                current_scpg_idx += 1
                if current_scpg_idx % 100 == 0 or current_scpg_idx == scpg_data_cnt:
                    trange_iteration.set_postfix_str(f'{current_scpg_idx}/{scpg_data_cnt} scpg data injected')
            else:
                mixed_index[idx] = f_in['index'][current_unsupervised_idx]
                current_unsupervised_idx += 1
        
        print(current_scpg_idx, scpg_data_cnt, current_unsupervised_idx, num_iterations)

        ## ========== inject complete ==========
        f_out_index_random.create_dataset('index', data=mixed_index)
        f_out_index_random.create_dataset('tasks', data=category_names)
        f_out_index_random.create_dataset('totals', data=totals)

        # convert samples ([task_idx_of_sample_1, task_idx_of_sample_2, task_idx_of_sample_3, ... task_idx_of_sample_N])
        # to index ([(task_idx_of_sample_1, data_idx_of_sample_1), (task_idx_of_sample_2, data_idx_of_sample_2), ...(task_idx_of_sample_N, data_idx_of_sample_N)])
        counters = [0] * num_categories
        index_sequence = []
        for idx, each in enumerate(samples):
            task_idx = int(each)
            index_sequence.append([task_idx, counters[task_idx]])
            counters[task_idx] += 1
        
        f_out_index_curriculum.create_dataset('tasks', data=category_names)
        f_out_index_curriculum.create_dataset('index', data=index_sequence)
        f_out_index_curriculum.create_dataset('totals', data=totals)

        print(f_in['tasks'][:])
        print(f_in['index'][:100])


def simulate_main(
    multitask_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5', 
    # scpg_file_path: str = '../data/ParaNMT50m_original/train_data_scpg.h5',
    curriculum_difficulty_file_path: str = './config/data_utils/curriculum_difficulties.json',
    curriculum_difficulty_setting_name: str = 'default',
    multitask_data_cnt_divisor: float = 30,
    scpg_data_cnt_divisor: float = 10,
    beta: float = 1.5,
    plot: bool = True,
    serialize: bool = False,
    export_directory: str = '../data/ParaNMT50m_original',
    export_name: str = 'train_data_multitask_small_with_scpg', 
    export_data: bool = False,
):
    totals, category_names = read_multitask_data(
        multitask_file_path=multitask_file_path,
        # scpg_file_path=scpg_file_path, 
        curriculum_difficulty_file_path=curriculum_difficulty_file_path,
        curriculum_difficulty_setting_name=curriculum_difficulty_setting_name,
        multitask_data_cnt_divisor=multitask_data_cnt_divisor,
        scpg_data_cnt_divisor=scpg_data_cnt_divisor
    )

    scale_sequences = get_scale_sequences_cosine(totals=totals, beta=beta)
    num_categories, num_iterations = scale_sequences.shape
    samples = np.zeros(num_iterations) # what category does the sample at timestep `i`` belongs to
    remainings = totals.copy()
    samples, percentages = simulate_numpy(samples=samples, remainings=remainings, scale_sequences=scale_sequences)
    if plot:
        plot_sample_cumsums(samples_arr=samples, category_names=category_names,)
        plot_sample_scales(scale_sequences=scale_sequences, category_names=category_names)
        plot_sample_scales(scale_sequences=percentages, category_names=category_names, savefig_filename='percentage.png')

    if serialize:
        serialize_curriculum_data(
            totals=totals,
            samples=samples,
            category_names=category_names,
            multitask_h5_file_path=multitask_file_path,
            # scpg_h5_file_path=scpg_file_path,
            export_directory=export_directory,
            export_name=export_name,
            export_data=export_data,
       )


def simulate_multi(
    multitask_file_path: str = '../data/ParaNMT50m_original/train_data_multitask.h5', 
    # scpg_file_path: str = '../data/ParaNMT50m_original/train_data_scpg.h5',
    curriculum_difficulty_file_path: str = './config/data_utils/curriculum_difficulties.json',
    curriculum_difficulty_setting_name: str = 'default',
    category_mask_file_path: str = './config/data_utils/category_masks.json',
    multitask_data_cnt_divisor: float = 30,
    scpg_data_cnt_divisor: float = 10,
    beta: float = 1.5,
    serialize: bool = False,
    export_directory: str = '../data/ParaNMT50m_original',
    export_name: str = 'train_data_multitask_small_with_scpg', 
):
    with open(category_mask_file_path, 'r') as f:
        category_masks = json.load(f)
    # check category mask validity
    assert (lens := len(set([*map(len, category_mask)]))) == 1, f'error, different lengths ({lens}) found in category masks'
    category_mask_sum = [0] * len(category_masks)
    for i in range(len(category_masks)):
        for j in range(len(category_masks[0])):
            category_mask_sum[j] += category_masks[i][j]
    assert (set(category_mask_sum) == {1}), f"error, category masks don't sum up to 1"
    # for every
    total_samples, total_percentages = [], []
    for mask_idx, category_mask in enumerate(category_masks):
        totals, category_names = read_multitask_data(
            multitask_file_path=multitask_file_path,
            # scpg_file_path=scpg_file_path, 
            curriculum_difficulty_file_path=curriculum_difficulty_file_path,
            curriculum_difficulty_setting_name=curriculum_difficulty_setting_name,
            cateogry_mask=category_mask,
            multitask_data_cnt_divisor=multitask_data_cnt_divisor,
            scpg_data_cnt_divisor=scpg_data_cnt_divisor
        )
        print(f"processing `{', '.join(category_names)}`...")
        scale_sequences = get_scale_sequences_cosine(totals=totals, beta=beta)
        num_categories, num_iterations = scale_sequences.shape
        samples = np.zeros(num_iterations) # what category does the sample at timestep `i`` belongs to
        remainings = totals.copy()
        samples, percentages = simulate_numpy(samples=samples, remainings=remainings, scale_sequences=scale_sequences)
        total_samples.append(samples)
        total_percentages.append(percentages)
    # plot_sample_cumsums(samples_arr=samples, category_names=category_names,)
    # plot_sample_scales(scale_sequences=scale_sequences, category_names=category_names)
    # plot_sample_scales(scale_sequences=percentages, category_names=category_names, savefig_filename='percentage.png')

    if serialize:
        serialize_curriculum_data(
            totals=totals,
            samples=samples,
            category_names=category_names,
            multitask_h5_file_path=multitask_file_path,
            # scpg_h5_file_path=scpg_file_path,
            export_directory=export_directory,
            export_name=export_name
       )


def split_index(index_file_path: str):
    pass

if __name__ == '__main__':
    simulate_main(
        multitask_file_path= '../data/ParaNMT50m_original/train_data_multitask.h5', 
        # scpg_file_path= '../data/ParaNMT50m_original/train_data_scpg.h5',
        curriculum_difficulty_file_path= './config/data_utils/curriculum_difficulties.json',
        curriculum_difficulty_setting_name= 'default',
        multitask_data_cnt_divisor= 1,
        scpg_data_cnt_divisor= 1,
        beta= 1.5,
        plot=True,
        serialize= True,
        export_directory= '../data/ParaNMT50m_original',
        export_name= 'train_data_multitask_all_data_with_full_scpg', 
    )