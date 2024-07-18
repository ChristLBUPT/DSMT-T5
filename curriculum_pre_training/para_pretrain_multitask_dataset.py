from torch.utils.data import Dataset, DataLoader
from data.TreeUtils.tree import MyTree
from data.TreeUtils.linearizing_tree import *
from tree_task_data_utils import TreeTaskFormatter
from transformers import PreTrainedTokenizer, BatchEncoding
from torch import distributed as dist
import os
import re
import logging
import h5py
import torch
from torch import Tensor, tensor
import random as rd
from typing import List, Tuple, Callable, Union, Dict, Literal, Type
from tqdm import tqdm
from itertools import chain
import json
import nltk
import subprocess

# =========================================================================================================
#                               task specific functions
# =========================================================================================================

def find_span_in_text(span: List[str], text: List[str]):
    for i in range(0, len(text) - len(span) + 1):
        if text[i: i + len(span)] == span:
            return [*range(i, i + len(span))]

    return -1
   
class TaskFormatter:
    @staticmethod
    def line_to_pruned_tree_parse(line: str):
        tree: MyTree = MyTree.fromstring(line)
        tree.get_height(1)
        tree = tree.prune_downwards(5, retain_children=MyTree.RetainChildren.ORIGINAL)
        sent = ' '.join(tree.leaves())
        return f"pruned tree parse: {sent}", tree.tostr_without_indent()
    
    # @staticmethod
    # def evaluating_pruned_tree_parse(result: str, label: str):
    #     try:
    #         t = MyTree.fromstring(result)
    #     except 

    @staticmethod
    def line_to_pruned_tree_completion(line: str):
        tree: MyTree = MyTree.fromstring(line)
        tree.get_height(1)
        template = tree.prune_downwards(5, retain_children=MyTree.RetainChildren.ORIGINAL)
        return f"pruned tree completion: {template.tostr_without_indent()}", tree.tostr_without_indent()

    @staticmethod
    def line_to_tree_pruning(line: str):
        tree: MyTree = MyTree.fromstring(line)
        tree.get_height(1)
        template = tree.prune_downwards(5, retain_children=MyTree.RetainChildren.ORIGINAL)
        return f"tree pruning: {tree.tostr_without_indent()}", template.tostr_without_indent()

    @staticmethod
    def line_to_constituency_searching(line):
        tree: MyTree = MyTree.fromstring(line)
        subtrees = [*tree.subtrees(lambda x: x.height() > 2)]
        subtree_labels = [each.label() for each in subtrees]
        rd.shuffle(subtree_labels)
        selected_label = subtree_labels[0]
        for idx, each in enumerate(subtree_labels):
            if subtree_labels.count(each) > 1:
                selected_label = each
                break
        
        all_spans = [' '.join(each.leaves()) for each in subtrees if each.label() == selected_label]
        sentence = ' '.join(tree.leaves())

        return f"constituency searching node: {selected_label} sentence: {sentence}", \
            " <sep> ".join(all_spans)
            # " ".join([each + " <sep>" for each in all_spans])

    @staticmethod
    def line_to_constituency_discrimination(line):
        tree: MyTree = MyTree.fromstring(line)
        subtrees = [*tree.subtrees(lambda x: x.height() > 2)]
        subtree_labels = [each.label() for each in subtrees]
        rd.shuffle(subtree_labels)
        selected_label = subtree_labels[0]
        is_negative_sample = rd.random() > 0.5
        for subtree in subtrees:
            if subtree.label() == selected_label: break
        
        span = subtree.leaves()
        sentence = tree.leaves()
        if is_negative_sample:
            span_idx = find_span_in_text(span, sentence)
            if span_idx == -1:
                raise ValueError(f"error, can't find span `{' '.join(span)}` in sentence `{' '.join(sentence)}`")
            span_idx = span_idx[0]
            span_end_idx = span_idx + len(span)
            random_offset = rd.randint(1, 3) * rd.choice([-1, 1])
            span_idx = min(max(span_idx + random_offset, 0), len(sentence))
            span_end_idx = max(min(span_end_idx + random_offset, len(sentence)), 0)
            span = sentence[span_idx:span_end_idx]

        return f"constituency discrimination node: {selected_label} span: {' '.join(span)} sentence: {' '.join(sentence)}", "False" if is_negative_sample else "True"

    @staticmethod
    def line_to_pos_tagging(line: str):
        tree: MyTree = MyTree.fromstring(line)
        token_and_pos_tags = " ".join(chain(*tree.pos()))
        return f"pos tagging: {' '.join(tree.leaves())}", token_and_pos_tags

    @staticmethod
    def line_to_production_detection(line: str):
        tree: MyTree = MyTree.fromstring(line)
        tree_with_leaf_masked = tree.prune_downwards(114514, retain_children=2, inference_height=True)
        productions = [str(each) for each in tree_with_leaf_masked.productions() if '<mask>' not in str(each)]
        # return f"production detection: {' '.join(tree.leaves())}", ' <sep> '.join(productions)
        return f"production detection: {line}", ' <sep> '.join(productions)
    
    
    tasks = ["pruned_tree_parse", "pruned_tree_completion", "tree_pruning", "constituency_searching", "constituency_discrimination", "pos_tagging", "production_detection"]

    tree_tasks = ["treeposition"]

    def get_method(self, task: str) -> Callable[[str], Tuple[str, str]]:
        return getattr(self, f"line_to_{task}")

class SupportsCollating:
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        padding_strategy: Literal['longest', 'max_length'] = "longest",
        max_length: int = 300,
        dont_collate: bool = False
        ):
        self.tokenizer = tokenizer
        self.dont_collate = dont_collate
        self.padding_strategy = padding_strategy
        self.max_length = max_length

    def collate_fn(self, batch):
        """
        collate of a batch of yielded/fetched tokenized lines
        Args:
            batch(List[Tuple[List[int]]]): A list of `[input sentence], [lable sentence]` pairs
        Returns:
            `BatchEncoding`: A huggingface transformer's BatchEncoding instance of the collated batch of inputs and labels
        """
        try:
            if not any(batch): return None
            if self.dont_collate: return batch
            src, label = zip(*batch)
            if self.padding_strategy == 'longest': # find the longest sentence in the batch, considering the `max_length`
                max_src_length = min(max([len(each) for each in src]), self.max_length)
                max_label_length = min(max([len(each) for each in label]), self.max_length)
            elif self.padding_strategy == 'max_length':
                max_src_length = self.max_length
                max_label_length = self.max_length
            attention_masks = []
            for src_sent in src: # padding and truncation
                while len(src_sent) > self.max_length: src_sent.pop(-1) # truncate sentences longer than `max_length`
                if len(src_sent) == self.max_length: src_sent[-1] = self.tokenizer.eos_token_id # replace the last token with <eos> token
                attention_masks.append([1 for _ in range(len(src_sent))] + [0 for _ in range(max_src_length - len(src_sent))]) # build attention mask
                src_sent += [self.tokenizer.pad_token_id] * (max_src_length - len(src_sent)) # padding src with <pad> token
            
            for label_sent in label:
                while len(label_sent) > self.max_length: label_sent.pop(-1) # truncate label to max_length
                if len(label_sent) == self.max_length: label_sent[-1] = self.tokenizer.eos_token_id # replace the last token with <eos> token
                label_sent += [-100] * (max_label_length - len(label_sent)) # padding label with -100

            return BatchEncoding({
                "input_ids": tensor(src),
                "attention_mask": tensor(attention_masks),
                "labels": tensor(label)
            })
        except:
            import debugpy; debugpy.breakpoint()

class ParaPretrainMultitaskDataset(Dataset, SupportsCollating):
    linearizing_formats = ["brackets", "backslash", "path"]
    linearizing_name2func = {
        "brackets": lambda x: x,
        "backslash": bracket_tree_to_slash,
        "path": bracket_to_path
    }
    linearizing_name2prefix = {
        "brackets": "syntax tree fill mask:",
        "backslash": "backslash syntax tree fill mask:",
        "path": "path syntax tree fill mask:"
    }
    def __init__(
        self, file_path: str, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 300, 
        padding_strategy: Literal['longest', 'max_length'] = "longest",
        mask_probability = 0.15,
        mask_span_average_length = 3,
        num_validation_samples: int = 35000,
        single_task: str = None,
        dont_collate: bool = False,
        # main_process_only: bool = False,
        ) -> None:
        """
        This is the dataset used to iterate over the ParaNMT50m dataset only consisting of constituency trees 
        and convert the trees to samples corresponding to multiple tasks
        Args:
            file_path: directory path which stores the hdf5 data file (a hdf5 file only consisting a dataset named `trees` which consists of string or bytes of linearized trees, such as `train_data.h5`)
            tokenizer: tokenizer instance (must be a huggingface PreTrainedTokenizer)
            max_length: max length of a single data instance 
            mask_probability: around how much proportion of tokens should be masked
            mask_span_average_length: the average length of continuous mask spans
            it's necessary to set this argument to True since only the master process will iterate this dataset
            single_task: whether only train on a specific task specified by this parameter, choices: [None, 'lo_intra_fillmask', 'hi_intra_fillmask', 'lo_inter_fillmask', 'hi_inter_fillmask'], 
            and those task names specified by `TaskFormatter`'s `tasks` list attribute
            dont_collate: whether to directly return the batch of tokenized samples when training (for the case that this dataset is used for caching data but not providing data during training)

        """
        super().__init__(tokenizer=tokenizer, max_length=max_length, padding_strategy=padding_strategy, dont_collate=dont_collate)
        self.file_path = file_path
        self.h5_file = h5py.File(self.file_path, 'r')
        self.data = self.h5_file['trees']
        logger_name = 'dataset'
        if dist.is_available() and dist.is_initialized():
            logger_name += f'_r{dist.get_rank()}_[{os.getpid()}]'
        self.logger = logging.getLogger(logger_name)
        self.num_validation_samples = num_validation_samples
        self.total_lines = len(self.data) - self.num_validation_samples

        # self.tokenizer = tokenizer
        # print(self.available_files)
        # self.max_length = max_length
        # self.padding_strategy = padding_strategy
        self.mask_probability = mask_probability
        self.original_mask_probability = mask_probability # probability reserved for possible altering of mask probability
        self.mask_span_average_length = mask_span_average_length
        self.task_formatter = TaskFormatter()
        self.single_task = single_task
        # self.dont_collate = dont_collate
        self.available_task_names = [None, 'lo_intra_fillmask', 'hi_intra_fillmask', 'lo_inter_fillmask', 'high_inter_fillmask'] + TaskFormatter.tasks
        assert self.single_task in self.available_task_names, f"error, single_task `{self.single_task}` not in available tasks ({', '.join(self.available_task_names)})"

        self.extra_ids = [self.tokenizer.convert_tokens_to_ids(f'<extra_id_{i}>') for i in range(100)]
        self.lb_idx, self.rb_idx = self.tokenizer.encode('()', add_special_tokens=False)

    # def get_t5_extra_id(self, extra_id: int):
    #     return self.tokenizer.convert_tokens_to_ids(f'<extra_id_{extra_id}>')

    def mask_intra_bracket(self, line: str, prefix: str):
        """mask contents inside brackets (maybe corresponding to syntax tree terminal nodes, pos tags and words)"""
        tokenized_prefix = self.tokenizer(prefix, add_special_tokens=False).input_ids
        tokenized_line = self.tokenizer(line, return_tensors='pt', add_special_tokens=False).input_ids[0]
        lb_idx, rb_idx = self.lb_idx, self.rb_idx
        bracket_mask = (tokenized_line == lb_idx).long() + (tokenized_line == rb_idx).long() # `1` for those positions corresponding to `▁(` or `)`
        # print(bracket_mask)
        # print([*zip(bracket_mask[0].tolist(), tok_with_syntax_terminals.convert_ids_to_tokens(tokenized_line[0]))])
        non_bracket_proportion = 1 - bracket_mask.sum().item() / bracket_mask.size(-1)
        mask_probability = self.mask_probability
        # real_mask_probability = mask_probability / non_bracket_proportion
        real_mask = torch.bernoulli((1 - bracket_mask) * mask_probability)

        target_token_ids = []
        extra_id_counter = 0

        for idx, mask in enumerate(real_mask.tolist()):
            if mask == 1 and (idx == 0 or real_mask[idx - 1] == 0):
                target_token_ids.extend([self.extra_ids[extra_id_counter], tokenized_line[idx].item()])
                tokenized_line[idx] = self.extra_ids[extra_id_counter]
                extra_id_counter += 1
            elif mask == 1:
                target_token_ids.append(tokenized_line[idx].item())
                tokenized_line[idx] = -1
        
        target_token_ids.append(self.extra_ids[extra_id_counter])
        
        tokenized_line = [each for each in tokenized_line.tolist() if each != -1]

        return tokenized_prefix + tokenized_line + [self.tokenizer.eos_token_id], target_token_ids + [self.tokenizer.eos_token_id]


    def process_line(self, line: str, prefix: str):
        """tokenize and mask a line (with continuous spans with average lenghth of `self.mask_span_average_length`), (alternatively) add prefix to a line, and returns it.
        Args:
            line: the line to process and return
        Returns:
            `List[int]`: list of (masked) tokenized token indices corresponding to `line`
        """
        tokenized_prefix = self.tokenizer(prefix, add_special_tokens=False).input_ids
        line = self.tokenizer(line, add_special_tokens=False).input_ids
        num_words = len(line)
        masks = (torch.rand(num_words) > (1 - self.mask_probability / self.mask_span_average_length)).long()  # 5% probability to become the `start` of mask span
        mask_lengths = torch.poisson(torch.ones(torch.sum(masks)) * self.mask_span_average_length)  # mask length follows Poisson(3)
        mask_positions = torch.nonzero(masks).view(-1)  # indices of mask starts
        for mask_idx, mask_pos in enumerate(mask_positions):
            if mask_lengths[mask_idx] == 0:
                masks[mask_pos] = -1  
                # `-1` stands for zero-length masks, that is, inserting a `<mask>` token into original sequence
            else:
                for j in range(mask_lengths[mask_idx].long().item()):
                    if mask_pos + j >= masks.size(-1): break
                    masks[mask_pos + j] = 1  # convert the following masks to 1
        
        try:
            line_masked = []  # replace masked part of original sentence to <extra_id_x> 
            masked_contents = []  # labels (<extra_id_0> masked content 1, <extra_id_1> masked content 2, ...)
            current_extra_idx = 0
            for i in range(len(line)):
                if masks[i] == 0 or current_extra_idx == 100:  # no mask / ignore masks more than 100
                    line_masked.append(line[i])
                elif masks[i] == -1:  # zero length mask (add `<mask>` before current token)
                    line_masked += [self.extra_ids[current_extra_idx], line[i]]
                    masked_contents.append(self.extra_ids[current_extra_idx])
                    current_extra_idx += 1
                else: # masks[i] == 1
                    if not line_masked or line_masked[-1] not in self.extra_ids: 
                        line_masked.append(self.extra_ids[current_extra_idx])
                        masked_contents += [self.extra_ids[current_extra_idx], line[i]]
                        current_extra_idx += 1
                        # only add `<mask>` when iterated to the first occurrence of consequential 1s
                        # and add a new sub-array to masked contents 
                    else:
                        masked_contents.append(line[i])
            

            return tokenized_prefix + line_masked \
                + [self.tokenizer.eos_token_id], masked_contents + [self.tokenizer.eos_token_id]
        except Exception as e:
            print(f'running status:\ni={i}, current_extra_idx={current_extra_idx}')
            print(f'data status:\nline: {line}\nmasks: {masks}\nline_masked: {line_masked}\nmasked_contents: {masked_contents}\nline_masked: {line_masked}\nmasked_contents: {masked_contents}\n')
            raise e
        # print('sent:', self.tokenizer.decode(line_masked), 'labels:', self.tokenizer.decode(masked_contents), sep='\n')
    
    def __getitem__(self, index):
        line = self.data[index].decode()
        if rd.random() < 0.6:  # 60 % probability of proceed mask filling for different linearizing formats
            fill_mask_task_prefix = rd.choice(['high difficulty intra span:', 'low difficulty intra span:', 'high difficulty inter span:', 'low difficulty inter span:'])
            if 'high' in fill_mask_task_prefix:
                self.mask_probability = self.original_mask_probability * 2
            else:
                self.mask_probability = self.original_mask_probability
            
            if 'intra' in fill_mask_task_prefix:
                return self.mask_intra_bracket(line, fill_mask_task_prefix)

            if 'inter' in fill_mask_task_prefix:
                return self.process_line(line, fill_mask_task_prefix)
            
            # if rd.random() > 0.5:
            #     self.mask_probability = self.original_mask_probability * 2  
            # else:
            #     self.mask_probability = self.original_mask_probability 
            # linearizing_format = rd.choices(self.linearizing_formats, weights=[5, 0.5, 0.5],)
            # linearizing_func = self.linearizing_name2func[linearizing_format]
            # converted_line = linearizing_func(line).strip()
            # selected_prefix = self.linearizing_name2prefix[linearizing_format]
            # return self.process_line(converted_line, )
        else:  # 40% probability of proceeding various syntax-intensive tasks
            task = rd.choice(self.task_formatter.tasks)
            input_text, tgt_text = self.task_formatter.get_method(task)(line)
            return self.tokenizer(input_text.strip()).input_ids, self.tokenizer(tgt_text.strip()).input_ids
    
    def __len__(self):
        return self.total_lines


    
    def split_dataset_by_task(self, proportion_dict: Dict[str, float] = None):
        """
        split the total dataset (h5py file) into task datasets according to their proportions.
        This is used to produce a uniformly splitted dataset for comparisons
        WARNING: This method is deprecated
        """
        pass

    
    def form_dataset_for_curriculum_learning(self, difficulty_dict: Dict[str, float] = None):
        """
        form a dataset using the curriculum learning paradigm (samples become more and more difficult )
        WARNING: This method is deprecated
        """
        pass

    def process_line_with_method(self, task_name: str):
        assert task_name in self.available_task_names, \
            f"error, single_task `{self.single_task}` not in available tasks ({', '.join(self.available_task_names)})"
        if 'fillmask' in task_name:
            if 'intra' in task_name: return self.mask_intra_bracket

# class ParaPretrainMultitaskTrainingDataset(Dataset):
#     def __init__(self, dataset_file: str, ) -> None:
#         super().__init__()

class ParaPretrainTreeMultitaskDataset(Dataset, SupportsCollating):
    def __init__(self,
        file_path: str,
        linearizing_format: Literal['bracket', 'slash'],
        tokenizer: PreTrainedTokenizer,
        padding_strategy: Literal['longest', 'max_length'] = "longest",
        max_length: int = 400,
        n_validation_examples: int = 40000,
        dont_collate: bool = False) -> None:
        super().__init__(tokenizer=tokenizer, padding_strategy=padding_strategy, max_length=max_length, dont_collate=dont_collate)
        self.data = h5py.File(file_path)['trees']
        self.task_formatter = TreeTaskFormatter()
        self.linearizing_format = linearizing_format
        self.num_data = len(self.data) - n_validation_examples

    def __getitem__(self, index):
        line = self.data[index]
        if not isinstance(line, str):
            line = line.decode()
        
        task_probabilities = [2, 1, 1, 1, 2]
        task = rd.choices(self.task_formatter.tasks, task_probabilities)[0]
        inputs, labels = self.task_formatter.get_method(task, self.linearizing_format)(line)
        return self.tokenizer(inputs.strip()).input_ids, self.tokenizer(labels.strip()).input_ids
    
    def __len__(self):
        return self.num_data
        

def build_val_set(file_path: str, num_samples: int, export_path: str, export_format: str = 'json', task_formatter_class: Union[TaskFormatter, TreeTaskFormatter] = TaskFormatter, linearize_format: Literal['bracket', 'slash'] = 'bracket'):
    """
    Turn a proportion of train data (specified by the `file_path` argument) into fixed pre-processed validation data and store them in a series of files
    Args:
        file_path: path to pre-training data
        num_samples: number of validation samples to produce
        export_path: path (directory) to export processed validation samples
        export_format: `json` or `txt`, txt format is for the convenience of EVALB evaluation (a common evaluation metrics for constituency parsing)
    """
    import json

    task_formatter = task_formatter_class()
    num_tasks = len(task_formatter.tasks)
    assert (num_samples % num_tasks) == 0, f"error, `num_samples` ({num_samples}) is not divisible by `num_tasks` ({num_tasks})"
    num_samples_per_task = num_samples // num_tasks
    
    with h5py.File(file_path, 'r') as f:
        raw_validation_samples = [each.decode() for each in f['trees'][-num_samples:]]
    for task_idx in range(num_tasks):
        task_samples = raw_validation_samples[task_idx * num_samples_per_task: (task_idx + 1) * num_samples_per_task]  # split raw data into tasks
        if export_format == 'json':
            processed_validation_samples = []
            for sample in task_samples:
                if isinstance(task_formatter, TaskFormatter):
                    processed_input, processed_target = task_formatter.get_method(task_formatter.tasks[task_idx])(sample)
                else:
                    processed_input, processed_target = task_formatter.get_method(task_formatter.tasks[task_idx], linearize_format)(sample)

                this_json_instance = {}
                this_json_instance["input"] = processed_input.strip()
                this_json_instance["target"] = processed_target.strip()
                this_json_instance["task_idx"] = task_idx
                processed_validation_samples.append(this_json_instance)
        
            with open(os.path.join(export_path, f"{task_idx:02d}_{task_formatter.tasks[task_idx]}.json"), 'w') as f:
                json.dump(processed_validation_samples, f, indent=2)
        elif export_format == 'txt' and task_idx in [0, 1, 2]:
            # processed_inputs, processed_targets = [], []
            processed_targets = []
            for sample in task_samples:
                _, processed_target = task_formatter.get_method(task_formatter.tasks[task_idx])(sample)
                # processed_inputs.append(MyTree.fromstring(processed_input).tostr_without_indent().strip())
                processed_targets.append(MyTree.fromstring(processed_target).tostr_without_indent().strip())
            
            # with open(os.path.join(export_path, f"{task_idx:02d}_{task_formatter.tasks[task_idx]}_inputs.txt"), 'w') as f:
            #     f.write('\n'.join(processed_inputs))
            with open(os.path.join(export_path, f"{task_idx:02d}_{task_formatter.tasks[task_idx]}.txt"), 'w') as f:
                f.write('\n'.join(processed_targets))


class ParaPretrainMultitaskValDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, included_tasks: List[str] = None) -> None:
        """
        Args:
            data_dir: directory consisting all evaluation json data files (xx-task_name.json)
            tokenizer: tokenizer used to tokenize data instances
        """
        super().__init__()
        self.data_dir = data_dir
        # self.task_formatter = TaskFormatter()
        self.data = []
        self.tokenizer = tokenizer
        self.tasks = {} # a integer-key dict mapping `task_idx` to `task_name`
        for name in filter(lambda x: x.endswith('.json'), os.listdir(data_dir)):
            task_idx, task_name = name.split("_", maxsplit=1)
            task_idx, task_name = int(task_idx), task_name.replace('.json', '')
            if included_tasks is None or task_name in included_tasks:
                self.tasks[task_idx] = task_name
        logging.info(f'Included tasks [{len(self.tasks)}]: {[*self.tasks.values()]}')

        for task_idx, task_name in self.tasks.items():
            task_instances = json.load(open(os.path.join(data_dir, f"{task_idx:02d}_{task_name}.json"), 'r'))
            for each in task_instances:
                each.update({"task_idx": task_idx, "task_name": task_name})
            self.data.extend(task_instances)
        # self.data.extend(json.load(open(os.path.join(data_dir, f"{task_idx + 1:02d}_scpg.json"), 'r')))
        # self.scpg_index = task_idx + 1

    def __getitem__(self, index):
        instance = self.data[index]
        return instance['input'], instance['target'], {"task": instance["task_name"], "task_idx": instance["task_idx"], "data_idx": index}
    
    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        # inputs: list of input strings, labels: list of label strings, metadata: list of metadata dicts
        inputs, labels, metadata = zip(*batch) 
        # tokenize input (tensor, with padding), labels (List[List[int]], without padding) and append metadata (task name and task index)
        return self.tokenizer(inputs, return_tensors='pt', padding=True), self.tokenizer(labels).input_ids, metadata
    
    # def evaluate(self, results: List[str])
    #     """accepts a list of evaluation results, consists of """


class ParaPretrainMultitaskCurriculumDataset(Dataset, SupportsCollating):
    def __init__(self, data_file_path: str, index_file_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 300, padding_strategy = "longest", task_mask: List[int] = None, subset_divisor: float = None, start_idx: int = None):
        """ 
        This is the dataset used for final training...
        Dataset the uses the index file specified by `index_file_path` to sample data from data file specified by `data_file_path`
        Args:
            data_file_path: path to the data h5 file (consisting to each task's individial data)
            index_file_path: path to the index h5 file (consisting to index ([task idx, sample idx] ... ), task names and totals)
            tokenizer: tokenizer used to tokenize the data 
            padding_strategy: strategy used to pad a batch of tokenized sequences
            task_mask: mask out samples from specific tasks (tasks corresponding to a `0` mask are masked out)
            subset_divisor: if specified, only use a subset of the dataset (round(total_len / `subset_divisor`))
            start_idx: CURRENTLY DEPRECATED
        """
        # print(super().__init__)
        # super().__init__()
        super().__init__(tokenizer=tokenizer, max_length=max_length, padding_strategy=padding_strategy, dont_collate=False)
        self.data_file_path = data_file_path
        self.index_file_path = index_file_path
        self.data_h5_file = h5py.File(self.data_file_path, 'r')
        self.index_h5_file = h5py.File(self.index_file_path, 'r')
        self.tasks = [(each if isinstance(each, str) else each.decode()) for each in self.index_h5_file['tasks'] ]
        self.task_mask = task_mask
        # assert not ((task_mask is not None) and (subset_divisor is not None)), "Cannot specify both `task_mask` and `subset_divisor`"
        self.real_index = self.index_h5_file['index'][...]
        self.total_len = len(self.real_index)
        if subset_divisor is not None:
            self.total_len = round(self.total_len / subset_divisor) if subset_divisor > 1 else round(self.total_len * subset_divisor)
            self.real_index = self.index_h5_file['index'][:self.total_len]
        if self.task_mask is None:
            self.total_len = len(self.real_index)
        else:
            self.real_index = [each for each in self.real_index if task_mask[each[0]] == 1]
            self.total_len = len(self.real_index)
        self.start_idx = start_idx
        self.subset_divisor = subset_divisor
        self.iterate_sequence = [*range(len(self.real_index))]
        # rd.shuffle(self.iterate_sequence)
        
        # if start_idx is not None:
        #     self.total_len -= start_idx
        # self.tokenizer = tokenizer
        # self.max_length = max_length
        # self.padding_strategy = padding_strategy
        # self.dont_collate = False # in order to fit the `ParaPretrainMultitaskDataset.collate_fn` method
        # self.task2num_samples = {each: self.data_h5_file[each].shape[-1] for each in self.tasks}
    
    def __getitem__(self, index):
        # logging.debug(f'fetched data [{index}]')
        if self.start_idx is not None and index < self.start_idx:
            return None
        # if self.task_mask is not None or self.subset_divisor is not None:
        task_idx, task_sample_idx = self.real_index[self.iterate_sequence[index]]
        return self.data_h5_file[self.tasks[task_idx]]['inputs'][task_sample_idx].tolist(), self.data_h5_file[self.tasks[task_idx]]['labels'][task_sample_idx].tolist()
        # else:
        #     task_idx, task_sample_idx = self.index_h5_file['index'][index]
        #     return self.data_h5_file[self.tasks[task_idx]]['inputs'][task_sample_idx].tolist(), self.data_h5_file[self.tasks[task_idx]]['labels'][task_sample_idx].tolist()
    
    def __len__(self):
        return self.total_len
    
    # collate_fn = ParaPretrainMultitaskDataset.collate_fn

def inspect_support_collate_dataset(
        dataset_class: Type, 
        *init_args, 
        tokenizer_path: str = '../pretrained-models/syntax-t5-base',
        stop_at: int = -1, 
        output_dir: str = '../outputs/debug_outputs/multi_task_dataset.out', 
        **init_kwargs,
    ):
    """inspect the data instances yielded by `ParaPretrainMultitaskCurriculumDataset`"""
    from transformers import T5Tokenizer
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('../pretrained-models/syntax-t5-base')
    dataset = dataset_class(*init_args, **init_kwargs)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=16,
        collate_fn=dataset.collate_fn)
    with open(output_dir, 'w') as f:
        for idx, batch in enumerate(tqdm(dataloader, total=stop_at if stop_at > 0 else None)):
            if stop_at > 0 and idx >= stop_at:
                break
            print('inputs:', file=f)
            # print(*tokenizer.batch_decode(batch.input_ids), sep='\n')
            print(tokenizer.decode(batch.input_ids[0]).replace('<pad>', ''), file=f)
            print('labels:', file=f)
            # print(*tokenizer.batch_decode([[(token_id if token_id != -100 else tokenizer.pad_token_id) for token_id in sample] for sample in batch.labels.tolist()]), sep='\n')
            print(tokenizer.decode([(idx if idx != -100 else tokenizer.pad_token_id) for idx in batch.labels[0]]).replace('<pad>', ''), file=f)
            if idx % 32 == 0:
                f.flush()


if __name__ == '__main__':

    # print(ParaPretrainMultitaskDataset.__mro__)
    # ===============================================================
    # from transformers import T5Tokenizer
    # from tqdm import tqdm
    # tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained('../pretrained-models/syntax-t5-base')
    # inspect_support_collate_dataset(ParaPretrainMultitaskCurriculumDataset, 
    #     '../data/ParaNMT50m_original/train_data_multitask_medium_with_scpg_data_sorted.h5',
    #     '../data/ParaNMT50m_original/train_data_multitask_medium_with_scpg_index_curriculum_mixed.h5' , tokenizer, stop_at=64, output_dir='../outputs/debug_outputs/multitask_dataset_curriculum.out')
    # inspect_support_collate_dataset(ParaPretrainMultitaskDataset, '../data/ParaNMT50m_original/train_data.h5', tokenizer, stop_at=64, output_dir='../outputs/debug_outputs/multitask_dataset.out')
    # ===============================================================
    # for export_format in ['txt', 'json']:
    # build_val_set("../data/ParaNMT50m_original/random_trees_<node>_bracket.h5", 25000, "../data/ParaNMT50m_original/val_data_bracket_tree/", 'json', TreeTaskFormatter, 'bracket')
    build_val_set("../data/ParaNMT50m_original/train_data.h5", 35000, "../data/ParaNMT50m_original/val_data/", 'json', TaskFormatter, 'bracket')
    # with open('./ParaNMT50m/val/src.txt')
    # ===============================================================
    # from transformers import T5Tokenizer
    # tok = T5Tokenizer.from_pretrained('./pretrained-models/t5-base')
    # ev = ParaPretrainMultitaskEvaluator("evaluation/apps/EVALB/evalb", "evaluation/apps/EVALB/sample/sample.prm", tok)
    # ev.evaluate_evalb('./data/ParaNMT50m_original/val_data/00_pruned_tree_parse.txt', './data/ParaNMT50m_original/val_data/00_pruned_tree_parse.txt')
    # ===============================================================
    # import svgling
    # tf = TaskFormatter()
    # with h5py.File("./data/ParaNMT50m_original/train_data.h5", 'r') as f:
    #     for idx, sent in enumerate(f['trees'][:300]):
    #         sent = sent.decode()
    #         svgling.draw_tree(MyTree.fromstring(sent)).get_svg().saveas(f"data/sample_trees/ParaNMT50m_original_filtered/{idx:03d}.svg")
    #         print('===' * 30)
    #         print(f"[{idx:03d}] sentence: {sent}")
    #         for task_name in tf.tasks:
    #             print(f"\n{task_name}:")
    #             processed_line, processed_target = tf.get_method(task_name)(sent)
    #             print("processed input:")
    #             print(processed_line)
    #             print("processed target:")
    # =========================================================================================================
    # from transformers import T5Tokenizer
    # tokenizer = T5Tokenizer.from_pretrained('../pretrained-models/syntax-t5-base-node-with-NT')
    # val_dataset = ParaPretrainMultitaskValDataset("../data/ParaNMT50m_original/val_data", tokenizer)
    # val_dataset_portion = ParaPretrainMultitaskValDataset("../data/ParaNMT50m_original/val_data", tokenizer, included_tasks=['pruned_tree_parse'])
    # print(len(val_dataset))
    # print(len(val_dataset_portion))