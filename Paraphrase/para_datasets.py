from typing import *
from typing import Union
# from openprompt.data_utils import InputExample
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler
from transformers import PreTrainedTokenizer, BatchEncoding, PhrasalConstraint
import sys; sys.path.append('..')
from data.data_processing import extract_tree_from_line
from data.TreeUtils.tree import MyTree
from nltk.tree import Tree
from transformers import PreTrainedTokenizer
from torch import tensor
from torch import distributed as dist
from tqdm import tqdm, trange
import multiprocessing
import os
import re
from copy import deepcopy
from itertools import chain
from collections import defaultdict
import logging
import pickle as pkl
from zss import simple_distance



tree_nonleaf_set = {'QP', 'LST', 'POS', 'NNP', 'RRC', 'NAC', 'VBD', 'VP', ':', 'DT', 'CD', 'WHADVP', ',', 'JJ', 'VBP',
                    'VBZ', 'SQ', 'CC', 'NP', 'ROOT', 'WP', 'RBS', 'FRAG', 'FW', 'PRN', 'EX', 'TO', 'WP$', 'WDT', 'IN',
                    'CONJP', 'SBAR', 'PRT', "''", 'LS', '.', 'PP', 'WHADJP', 'PDT', 'MD', 'VBN', 'PRP', 'S', 'RB',
                    'ADJP', 'VBG', 'SYM', 'UH', 'WRB', 'NNS', 'X', 'PRP$', 'JJR', 'WHPP', 'NNPS', 'VB', 'NN', 'SBARQ',
                    'JJS', 'INTJ', 'RP', 'RRB', 'SINV', 'UCP', 'RBR', '``', 'LRB', 'ADVP', 'NX', 'WHNP'}

def preprocess(x: Union[str, MyTree], add_bracket_spaces: bool = False) -> str:
    """Add spaces to brackets (optional) and remove `@@ `"""
    if isinstance(x, MyTree):
        x = x.tostr_without_indent()
    if add_bracket_spaces:
        x = x.replace('(', ' ( ').replace(')', ' ) ')
        x = re.sub(' +', ' ', x)
    x = x.replace('@@ ', '')
    return x


def pad_and_mask(seqs: List[List[int]], pad_id: int, max_length: int):
    max_len = min(max([len(each) for each in seqs]), max_length)
    res = []
    masks = []
    for each in seqs:
        if len(each) > max_length:
            each = each[:max_length]
        res.append(each + [pad_id] * (max_len - len(each)))
        masks.append([1 for _ in range(len(each))] + [0 for _ in range(max_len - len(each))])
    return res, masks
    
def replace_mask_with_extra_id(seq: str):
    """replace <mask> with <extra_id_0>, <extra_id_1>, ..."""
    seq_split_by_mask = seq.split('<mask>')
    extra_ids = [f'<extra_id_{i}>' for i in range(len(seq_split_by_mask))]
    return ''.join(chain(*zip(seq_split_by_mask, extra_ids)))[:-len(extra_ids[-1])]

def replace_mask_with_glm_mask_token(seq: str):
    return seq.replace('<mask>', '[MASK]')
    
def get_all_children(tree: MyTree):
    """get all children of a tree"""
    res = []
    for each in tree:
        if isinstance(each, MyTree):
            res.extend(get_all_children(each))
        else:
            res.append(each)
    return res

def wrap_children_with_extra_ids(tree: MyTree):
    """wrap all children of a tree with <extra_id_0>, <extra_id_1>, ..."""
    children = get_all_children(tree)
    extra_ids = [f'<extra_id_{i}>' for i in range(len(children) + 1)]
    return ''.join(chain(*zip(extra_ids[:-1], children))) + extra_ids[-1]


def strip_children(t: Tree):
    new_tree = Tree(t.label(), [])
    for each in t:
        if isinstance(each, Tree):
            new_tree.append(strip_children(each))
    
    return new_tree

def get_compared_tree(tgt_tree: Tree, ref_tree: Tree, return_leaf_pruned_tree: bool = False):
    tgt_tree, ref_tree = map(strip_children, (tgt_tree, ref_tree))
    dist, ops = simple_distance(ref_tree, tgt_tree, lambda x: [*x], Tree.label, return_operations=True)
    matched_trees = []
    # find all matched nodes
    for each in ops:
        if each.type == each.match:
            matched_trees.append(each.arg2)    
    # make all non-match nodes wrapped in `**` in a new tree 
    compared_tree = deepcopy(tgt_tree)
    for pos_seq in tgt_tree.treepositions():
        current_node = tgt_tree[pos_seq]
        if current_node not in matched_trees:
            compared_tree[pos_seq]._label = f'*{compared_tree[pos_seq].label()}*'
    
    return compared_tree if not return_leaf_pruned_tree else (compared_tree, tgt_tree, ref_tree)
    

# class ParaNMTDemoDataset(Dataset):
#     def __init__(self, data_dir: str, has_ref: bool):
#         self.has_ref = has_ref
#         self.src_texts = [each.strip() for each in open(os.path.join(data_dir, 'src.txt'))]
#         self.tgt_texts = [each.strip() for each in open(os.path.join(data_dir, 'tgt.txt'))]
#         self.num_texts = len(self.src_texts)
#         self.current_iter = 0

#     def __getitem__(self, item):
#         return self.src_texts[item], self.tgt_texts[item]

#     def __iter__(self):
#         self.current_iter = 0
#         return self

#     def __next__(self):
#         if self.current_iter < self.num_texts:
#             res = InputExample(
#                 guid=self.current_iter,
#                 tgt_text=self.tgt_texts[self.current_iter],
#                 meta={
#                     'src_text': self.src_texts[self.current_iter]
#                 }
#             )
#             self.current_iter += 1
#             return res
#         else:
#             raise StopIteration

#     def __len__(self):
#         return self.num_texts

class ParaNMTDataset(Dataset):
    """
    Args: 
    data_dir: directory of your dataset
    model: model TYPE ("t5" or "bart")
    prune_height: height you want to prune your tree(not including root node)
    has_ref: whether or not the data has reference sentences(depending on data split)
    use_num_postfix: where or not to use number postfix such as `VP-1`, `NP-2`, etc
    tgt_only: only available for BART model data (whether or not let model predict src text and src parse)
    use_tgt_as_ref: during validation/test, whether to use the TEMPLATE syntax tree as syntactic reference
    instance_format_name: format of the instance, can be one of `<xxx_tree>`, `t5_prefix`, `<tree>`, format shown as follows:
        <xxx_tree>: <src> [source sentence] <src_t> [linerized source tree] <temp_t> [(masked) linearized template/target tree]
        t5_prefix: scpg source sentence: [source sentence] source syntax tree: [linerized source tree] template syntax tree: [(masked) linearized template/target tree]
        <tree>: <sent> [source tree] <tree> [linerized source tree] <tree> [(masked) linearized template/target tree]
    mask: whether mask leaves of target trees, default to `True`, false for contrasive baselines
      
    `NOTE: if you're using SGCP's source code, please notice that in terms of `tgt` and `ref`, our data is inversed
    """
    def __init__(self, data_dir: str, 
            model: str, tokenizer: PreTrainedTokenizer, 
            prune_height: int, 
            has_ref: bool, 
            use_num_postfix: bool = False, 
            tgt_only: bool = False,
            use_tgt_as_ref: bool = False,
            add_bracket_spaces: bool = False,
            instance_format_name: str = 'src_tgt',
            mask: bool = True,
            device: str = None,
            max_length: int = 1 << 30,
            trunc_size: Union[int, None] = None
        ):
        self.data_dir = data_dir
        self.model = model
        self.tokenizer = tokenizer
        self.prune_height = prune_height
        self.has_ref = has_ref
        self.trunc_size = trunc_size
        self.max_length = max_length
        self.tgt_only = tgt_only
        self.use_tgt_as_ref = use_tgt_as_ref
        self.add_bracket_spaces = add_bracket_spaces
        self.instance_format_name = instance_format_name
        # the third part, corresponding to template tree (if val/test dataset and use template data) or target tree
        part3 = '{ref_tree_masked}' if self.has_ref and (not self.use_tgt_as_ref) else '{tgt_tree_masked}'
        self.instance_formats = {
            # mask-based formats
            '<xxx_tree>': '<src> {src_text} <src_t> {src_tree} <temp_t> ' + part3,
            't5_prefix': 'scpg source sentence: {src_text} source syntax tree: {src_tree} template syntax tree:' + part3,
            '<tree>': '<sent> {src_text} <tree> {src_tree} <tree> ' + part3,
            # span-based formats
            'aesop': '{src_text} <sep> {src_tree} <sep> ' + ('{ref_tree_noleaf}' if self.has_ref and (not self.use_tgt_as_ref) else '{tgt_tree_noleaf}'),
            'aesop_nooutputtree': '{src_text} <sep> {src_tree} <sep> ' + ('{ref_tree_noleaf}' if self.has_ref and (not self.use_tgt_as_ref) else '{tgt_tree_noleaf}'),
            # triplet (tree-free) based formats
            'triplet': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_nooutputtree': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_token': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_tgtonly':  'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_tree': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_tgttreeonly': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_sentencetree': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            # 'triplet_sentencetgttree': '{src_text} <sep> {ref_text}',
            'triplet_sentencetgttree': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_commontree': 'no-tree-scpg source sentence: {src_text} template sentence: {ref_text}',
            'triplet_treesentencetoken': '<ref> {ref_text} <src> {src_text}'
        }
        masked_method_label_format = '{tgt_text}' if self.has_ref else '{masked_values}'
        self.instance_label_formats = {
            # mask-based formats
            '<xxx_tree>': masked_method_label_format,
            't5_prefix': masked_method_label_format,
            '<tree>': masked_method_label_format,
            # span-based formats
            'aesop': ('{ref_tree_noleaf}' if self.has_ref and (not self.use_tgt_as_ref) else '{tgt_tree_noleaf}') + ' <sep> {tgt_text}',
            'aesop_nooutputtree': '{tgt_text}',
            # triplet (tree-free) based formats
            'triplet': 'source syntax tree: {src_tree} template syntax tree: {ref_tree} target sentence: {tgt_text}',
            # 'triplet_nooutputtree': 'target sentence: {tgt_text}',
            'triplet_nooutputtree': '{tgt_text}',
            'triplet_token': '<src_tree> {src_tree} <temp_tree> {ref_tree} <tgt> {tgt_text}',
            'triplet_tgtonly': '{ref_tree} <tgt> {tgt_text}',
            'triplet_tree': '{ref_tree} <tgt> {tgt_tree}',
            'triplet_tgttreeonly': '{tgt_tree}',
            'triplet_sentencetree': '{tgt_text} <tgt> {ref_tree}',
            'triplet_sentencetgttree': '{tgt_text} <tgt> {tgt_tree}',
            'triplet_commontree': '{tgt_text} <tgt> {common_tree}',
            'triplet_treesentencetoken': '<ref> {ref_tree} <src> {src_tree} <tgt> {tgt_text}'
        }
        self.instance_format_name2extract_method = { 
            # this is used to extract target sentence from model outputs during evaluation
            # the value corresponds to `extract_method`
            "<xxx_tree>": "extra_id",
            "t5_prefix": "extra_id",
            "<tree>": "extra_id",
            "aesop": "after_`<sep>`",
            "aesop_nooutputtree": "direct",
            "triplet": "after_`target sentence:`",
            # "triplet_nooutputtree": "after_`target sentence:`",
            "triplet_nooutputtree": "direct",
            "triplet_token": "after_`<tgt>`",
            "triplet_tgtonly": "after_`<tgt>`",
            "triplet_tree": "triplet_tree",
            "triplet_parse": "direct",
            'triplet_tgttreeonly': "leaf",
            'triplet_sentencetgttree': "sentence_tgt_tree",
            "triplet_sentencetree": "sentence_tree",
            "triplet_commontree": "common_tree",
        }
        self.instance_format2constraint = defaultdict(list)
        self.instance_format2constraint.update({
            # "<xxx_tree>": [],
            # "t5_prefix": [],
            # "<tree>": [],
            "aesop": [PhrasalConstraint(tokenizer("<sep>", add_special_tokens=False).input_ids)],
            # "aesop_nooutputtree": [],
            "triplet": [
                PhrasalConstraint(tokenizer("source syntax tree:", add_special_tokens=False).input_ids),
                PhrasalConstraint(tokenizer("template syntax tree:", add_special_tokens=False).input_ids),
                PhrasalConstraint(tokenizer("target sentence:", add_special_tokens=False).input_ids),
            ],
            # "triplet_nooutputtree": [],
            # "triplet_token": [],
            # "triplet_tgtonly": [],
            # "triplet_tree": [],
            # "triplet_parse": [],
            # "triplet_tgttreeonly": [],
            # "triplet_sentencetgttree": [],
            # "triplet_sentencetree": [],
            # "triplet_commontree": []
        })
        self.mask = mask
        self.device = device
        self.logger = logging.getLogger(__name__ + f'_r{dist.get_rank()}' if dist.is_initialized() else '')
        need_tqdm = dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized()
        assert self.model in ['bart', 't5', 'glm'], f"Error: unsupported model type `{model}`"
        if self.instance_format_name == 'triplet':
            assert self.has_ref, f'`triplet` instance format must have `has_ref` attribute set to `True`'
        # self.logger.info('Start creating instances...')
        # first inspect cache, if cache not exists, read sentences and trees, and create them, else read from cache
        cache_path = os.path.join(self.data_dir, f'cached_data_t5_p{self.prune_height}')#_{"tgt" if self.use_tgt_as_ref else "temp"}')
        if not os.path.exists(cache_path):
            self.logger.info('Cache path not exists, try reading data file and forming instances...')
            # texts/trees: {'src': [source texts/trees], 'tgt': [target texts/trees]}
            self.texts = defaultdict(list)
            self.trees = defaultdict(list)
            for split in ['src', 'tgt', 'ref']:
                self.logger.info(f'Split {split}')
                if split == 'ref' and not self.has_ref: continue
                # has_ref corresponds to val/test sets, which is rather small, with no need for progress bars
                if self.has_ref: need_tqdm = False 
                commented_indices = set() # regard `#` as commented sentences
                self.logger.info(f'Reading texts...')
                with open(os.path.join(data_dir, f'{split}.txt')) as f:
                    sent_list = f.read().split('\n')
                    if need_tqdm: sent_list = tqdm(sent_list)
                    for idx, sent in enumerate(sent_list):
                        if sent[0] == '#':
                            commented_indices.add(idx)
                        else:
                            self.texts[split].append(sent)

                if os.path.exists(os.path.join(data_dir, f'{split}_trees.pkl')):
                    self.logger.info(f'Extracting trees from {split}_trees.pkl')
                    ## read trees from pickle file and ignore commented sentences
                    self.trees[split] = [each for idx, each in \
                        enumerate(pkl.load(open(os.path.join(data_dir, f'{split}_trees.pkl'), 'rb'))) if idx not in commented_indices]
                    if self.trees[split][0].my_height == -1: # calculate heights
                        for each in self.trees[split]:
                            each.get_height(1)
                        with multiprocessing.Lock():
                            pkl.dump(self.trees[split], open(os.path.join(data_dir, f'{split}_trees.pkl'), 'wb'))
                        
                else: # extract trees from xxx.txt-corenlp-opti
                    self.logger.info(f'Reading trees from {split}.txt-corenlp-opti')
                    with open(os.path.join(data_dir, f'{split}.txt-corenlp-opti')) as f:
                        tree_list = f.read().split('\n')
                        if need_tqdm: tree_list = tqdm(tree_list)
                        for idx, each in enumerate(tree_list): # for each line, extract tree and calculate heights
                            if each.startswith('{') and idx not in commented_indices:
                                tree = extract_tree_from_line(each, not use_num_postfix)
                                tree.get_height(1)
                                self.trees[split].append(tree)

                    if not commented_indices: # serialize extracted trees to pickle file
                        self.logger.info(f'Serializing trees to {split}_trees.pkl')
                        with multiprocessing.Lock():
                            pkl.dump(self.trees[split], open(os.path.join(data_dir, f'{split}_trees.pkl'), 'wb'))
            
            # self.logger.info('Split tgt')
            # with open(os.path.join(data_dir, 'tgt.txt')) as f:
            #     self.tgt_texts = [each for each in f.read().split('\n') if each and each[0] != '#']
            # with open(os.path.join(data_dir, 'tgt.txt-corenlp-opti')) as f:
            #     self.tgt_trees = [extract_tree_from_line(each, not use_num_postfix) \
            #         for each in (tqdm(f.read().split('\n')) if need_tqdm else f.read().split('\n')) \
            #         if each.startswith('{')]
            #     for each in self.tgt_trees:
            #         each.get_height(1)

            # self.logger.info('Split ref')
            # if has_ref:
            #     with open(os.path.join(data_dir, 'ref.txt')) as f:
            #         self.ref_texts = [each for each in f.read().split('\n') if each and each[0] != '#']
            #     with open(os.path.join(data_dir, 'ref.txt-corenlp-opti')) as f:
            #         self.ref_trees = [extract_tree_from_line(each, not use_num_postfix) for each in f.read().split('\n') if each.startswith('{')]
            #         for each in self.ref_trees:
            #             each.get_height(1)

            self.src_texts, self.tgt_texts, self.ref_texts = self.texts['src'], self.texts['tgt'], self.texts['ref']
            self.src_trees, self.tgt_trees, self.ref_trees = self.trees['src'], self.trees['tgt'], self.trees['ref']

            assert len(self.src_texts) == len(self.src_trees), f'source text file length ({len(self.src_texts)}) and tree file length ({len(self.src_trees)}) must be equal'
            self.num_texts = len(self.src_texts)
            self.logger.info('creating instances...') # create instances
            self.instances = [self._form_instance(i) for i in (trange(self.num_texts) if need_tqdm else range(self.num_texts))]
            with multiprocessing.Lock():
                self.logger.info(f'caching instances to {cache_path}')
                pkl.dump(self.instances, open(cache_path, 'wb'))
        else:
            self.logger.info(f'loading cached instance from {cache_path}...')
            self.instances = pkl.load(open(cache_path, 'rb'))
            self.num_texts = len(self.instances)


    def _form_instance(self, idx: int):
        """Form an instance (a dictionary containing corresponding src and tgt texts and pruned (maybe masked) trees) for a single example"""
        if self.model == 'bart':
            # ====================================================================================================================================
            #                               BART is now deprecated
            # inst = {
            #     "src_text": self.src_texts[idx],
            #     "src_tree": preprocess(self.src_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
            #     "tgt_text": self.tgt_texts[idx],
            # }
            # if self.has_ref:
            #     inst.update({
            #         "ref_tree_masked": preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 2), self.add_bracket_spaces),
            #         # "ref_tree": preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 1)),
            #     })
            # else:
            #     inst.update({
            #         "tgt_tree_masked": preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 2), self.add_bracket_spaces),
            #         "tgt_tree": preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
            #     })
            # return inst
            raise NotImplementedError
        elif self.model == 't5' or self.model == 'glm':
            try:
                inst = {
                    # source (only sentence and pruned (with leaf) tree)
                    "src_text": self.src_texts[idx],
                    "src_tree": preprocess(self.src_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                    # target (sentence, pruned (with leaf, without leaf, masked) tree)
                    "tgt_text": self.tgt_texts[idx],
                    "tgt_tree": preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                    "tgt_tree_noleaf": preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 0), self.add_bracket_spaces),
                    "tgt_tree_masked": (replace_mask_with_extra_id if self.model == 't5' else replace_mask_with_glm_mask_token)\
                        (preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 2), self.add_bracket_spaces)),
                }
            except IndexError:
                print(f'index: {idx}, total: {len(self)}')
                breakpoint()
            if self.has_ref:
                inst.update({
                    # [Optional] target (sentence, pruned (with leaf, without leaf, masked) tree)
                    "ref_text": self.ref_texts[idx],
                    "ref_tree": preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                    "ref_tree_noleaf": preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 0), self.add_bracket_spaces),
                    "ref_tree_masked": (replace_mask_with_extra_id if self.model == 't5' else replace_mask_with_glm_mask_token)\
                        (preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 2), self.add_bracket_spaces)),
                })
            else:
                inst.update({
                    "masked_values": preprocess(wrap_children_with_extra_ids(self.tgt_trees[idx].prune_downwards(self.prune_height, 1)), self.add_bracket_spaces)
                })
            return inst

    def create_instances(self):
        """create instances\n
        A single instance consists of follows\n
            for `train` split: (input_ids, label_ids), 
            for `val/test` split: (input_ids, tgt_text))
        """
        pass
        # results = []
            
        # return results

    def __iter__(self):
        self.current_iter = 0
        # while self.current_iter < len(self):

    def __getitem__(self, item):
        # if isinstance(item, int):
        #     indices = [item]
        # else:
        #     indices = list(range(*item))
        # for item in trange(self.num_texts, desc="Tokenizing and Creating instances..."):
        # if self.model == 'bart':
            # ==========WARNING: BART IS NOW DEPRECATED==========
            # if self.has_ref:
            #     inst_text = f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['ref_tree_masked']} <temp> <mask>"
            #     label_text = inst['tgt_text']
            #     return self.tokenizer([inst_text]).input_ids[0], label_text
            #     # return InputExample(
            #     #     text_a=self.src_texts[item], text_b=self.ref_texts[item], tgt_text=self.tgt_texts[item]
            #     # )
            # else:
            #     inst_text = f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['tgt_tree_masked']} <temp> <mask>"
            #     label_text = f"<temp_t> {inst['tgt_tree']} <temp> {inst['tgt_text']}" if self.tgt_only else \
            #         f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['tgt_tree']} <temp> {inst['tgt_text']}"
            #     # label_text = f"<temp_t> {inst['tgt_tree']} <temp> {inst['tgt_text']}" if self.tgt_only else \
            #     #     f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['tgt_tree']} <temp> {inst['tgt_text']}"
            #     return self.tokenizer([inst_text]).input_ids[0], self.tokenizer([label_text]).input_ids[0]
            # ==================================================
            # pass
            # return InputExample(
            #     text_a=self.src_texts[item], text_b=self.tgt_texts[item], tgt_text=self.tgt_labels[item]
            # )
        if self.model in ['t5', 'bart']:
            if self.instance_format_name != 'triplet_parse':
                inst = self.instances[item]
                instance_format = self.instance_formats[self.instance_format_name]
                label_format = self.instance_label_formats[self.instance_format_name]
                inst_text = instance_format.format(**inst)
                label_text = label_format.format(**inst)
            else:
                inst = self.instances[item // 2]
                inst_text = f'pruned tree parse: {inst["ref_text"] if item & 1 else inst["src_text"]}'
                label_text = inst["ref_tree"] if item & 1 else inst["src_tree"]

            if item % 50 == 0:
                self.logger.debug(f"idx: [{item}], inputs: `{inst_text}`, label: `{label_text}`")
                # self.logger.debug(self.instance_formats[self.instance_format_name])
                # self.logger.debug(inst)
            # return self.tokenizer([inst_text]).input_ids[0], label_text if self.has_ref else self.tokenizer([label_text]).input_ids[0]
            return self.tokenizer([inst_text]).input_ids[0], self.tokenizer([label_text]).input_ids[0]
            # if self.has_ref: # for validation and test sets
            #     if self.use_tgt_as_ref:
            #         inst_text = instance_format.format(inst['src_text'], inst['src_tree'], inst['tgt_tree_masked'])
            #         # inst_text = f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['tgt_tree_masked']}" 
            #     else:
            #         inst_text = instance_format.format(inst['src_text'], inst['src_tree'], inst['ref_tree_masked'])
            #         # inst_text = f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['ref_tree_masked']}" 
            #     label_text = inst['tgt_text']
            #     self.logger.debug(f"\n\tinstance: {inst_text},\n\tlabel: {label_text}")
            #     return self.tokenizer([inst_text]).input_ids[0], label_text
            # else:  # training set
            #     inst_text = instance_format.format(inst['src_text'], inst['src_tree'], inst['tgt_tree_masked'])
            #     # inst_text = f"<src> {inst['src_text']} <src_t> {inst['src_tree']} <temp_t> {inst['tgt_tree_masked']}"
            #     label_text = inst['masked_values']
            #     self.logger.debug(f"\n\tinstance: {inst_text},\n\tlabel: {label_text}")
            #     return self.tokenizer([inst_text]).input_ids[0], self.tokenizer([label_text]).input_ids[0]
        else:
            raise NotImplementedError(f"Unsupported model type `{self.model}`")
        # return self.instances[item]
    
    def collate_fn(self, batch):
        input_ids, attention_mask = pad_and_mask([each[0] for each in batch], self.tokenizer.pad_token_id, self.max_length)
        if isinstance(batch[0][1], str):
            labels = None
        else:
            labels, _ = pad_and_mask([each[1] for each in batch], -100, self.max_length)
        batch_encoding = BatchEncoding({"input_ids": tensor(input_ids), "attention_mask": tensor(attention_mask)})
        if labels is not None: batch_encoding["labels"] = tensor(labels)
        return batch_encoding if self.device is None else batch_encoding.to(self.device)
        

    # def __next__(self):
    #     if self.current_iter == self.num_texts or self.trunc_size is not None and self.current_iter == self.trunc_size:
    #         raise StopIteration

    #     if self.has_ref:
    #         res = InputExample(
    #             # guid=self.current_iter,
    #             text_a=self.src_trees[self.current_iter],
    #             text_b=self.ref_trees[self.current_iter],
    #             tgt_text=self.tgt_texts[self.current_iter]
    #         )
    #         self.current_iter += 1

    #     else:
    #         res = InputExample(
    #             # guid=self.current_iter,
    #             text_a=self.src_trees[self.current_iter],
    #             text_b=self.tgt_trees[self.current_iter],
    #             tgt_text=self.tgt_labels[self.current_iter]
    #         )
    #         self.current_iter += 1

    #     return res

    def __len__(self):
        return self.num_texts if self.instance_format_name != 'triplet_parse' else (2 * self.num_texts)

class ParaNMTTripleDataset(ParaNMTDataset):
    def __init__(self, data_dir: str, 
            model: str, tokenizer: PreTrainedTokenizer, 
            prune_height: int, 
            use_num_postfix: bool = False, 
            add_bracket_spaces: bool = False,
            device: str = None,
            max_length: int = 1 << 30,
            trunc_size: Union[int, None] = None
        ):
        super().__init__(
            data_dir=data_dir,
            model=model,
            tokenizer=tokenizer,
            prune_height=prune_height,
            has_ref=True,
            use_num_postfix=use_num_postfix,
            tgt_only=False,
            use_tgt_as_ref=False,
            add_bracket_spaces=add_bracket_spaces,
            instance_format_name='t5_prefix',
            device=device,
            max_length=max_length, 
            trunc_size=trunc_size)
        self.instance_format = 'no-tree-scpg source sentence: {0} template sentence: {1}' # only support 't5_prefix' style formats
        self.label_format = 'source syntax tree: {0} template syntax tree: {1} target syntax tree: {2}'
        
    def __getitem__(self, item):
        inst = self.instances[item]
        if self.model == 't5':
            inst_text = self.instance_format.format(inst['src_text'], inst['ref_text'])
            label_text = self.label_format.format(inst['src_tree'], inst['ref_tree'], inst['tgt_tree'])
            if item % 50 == 0:
                self.logger.debug(f"\n\tinstance: {inst_text},\n\tlabel: {label_text}")
            return self.tokenizer([inst_text]).input_ids[0], self.tokenizer([label_text]).input_ids[0]
        else:
            raise NotImplementedError(f"Unsupported model type `{self.model}`")
    
    def _form_instance(self, idx: int):
        """Form an instance (a dictionary containing corresponding src and tgt texts and pruned (maybe masked) trees) for a single example"""
        if self.model == 't5' or self.model == 'glm':
            try:
                inst = {
                    "src_text": self.src_texts[idx],
                    "src_tree": preprocess(self.src_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                    "ref_text": self.ref_texts[idx],
                    "ref_tree": preprocess(self.ref_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                    "tgt_text": self.tgt_texts[idx],
                    "tgt_tree": preprocess(self.tgt_trees[idx].prune_downwards(self.prune_height, 1), self.add_bracket_spaces),
                }
            except IndexError:
                print(f'index: {idx}, total: {len(self)}')
                breakpoint()
            return inst


class BlockDistributedSampler(Sampler):
    def __init__(self, data_source: Optional[Sized], rank: int = None, world_size: int = None) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.world_size = dist.get_world_size() if world_size is not None else world_size
        self.rank = dist.get_rank() if rank is not None else rank
        indices = [*range(len(data_source))]
        block_size = len(data_source) // self.world_size
        self.indices = indices[self.rank * block_size: (self.rank + 1) * block_size] if self.rank != self.world_size - 1 \
            else indices[self.rank * block_size:]

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __len__(self):
        return len(self.data_source)

# class ParaNMTDataset(Dataset):
#     """
#     params: 
#     data_dir: directory of your dataset
#     model: model type ("t5" or "bart")
#     prune_height: height you want to prune your tree(not including root node)
#     has_ref: whether or not the data has reference sentences(depending on data split)
#     """
#     def __init__(self, data_dir: str, model: str, prune_height: int, has_ref: bool,
#                  trunc_size: Union[int, None] = None, tokenizer: Union[PreTrainedTokenizer, None] = None):
#         self.prune_height = prune_height
#         self.has_ref = has_ref
#         self.model = model
#         self.src_trees = [preprocess(each.strip()) for each in open(
#             os.path.join(data_dir, model, f'src.txt-with-tree-p{self.prune_height}-original'))]
#         if self.has_ref:
#             self.ref_trees = [preprocess(each.strip()) for each in open(
#                 os.path.join(data_dir, model, f"ref.txt-with-tree-p{self.prune_height}-mask"))]
#             self.tgt_texts = [each.strip() for each in open(os.path.join(data_dir, f"tgt.txt"))]

#         else:
#             self.tgt_trees = [preprocess(each.strip()) for each in open(
#                 os.path.join(data_dir, model, f"tgt.txt-with-tree-p{self.prune_height}-mask"))]
#             self.tgt_labels = [preprocess(each.strip()) for each in open(
#                 os.path.join(data_dir, model, f"tgt.txt-with-tree-p{self.prune_height}-label"))]
#         self.num_texts = len(self.src_trees)
#         self.trunc_size = trunc_size
#         self.eos_token = tokenizer.eos_token

#     def __iter__(self):
#         self.current_iter = 0
#         return self

#     def __getitem__(self, item):
#         # if isinstance(item, int):
#         #     indices = [item]
#         # else:
#         #     indices = list(range(*item))
#         if self.model == 'bart':
#             if self.has_ref:
#                 return f"<src_t> {self.src_trees[item]} <temp_t> {self.ref_trees[item]}", self.tgt_texts[item]
#                 # return InputExample(
#                 #     text_a=self.src_texts[item], text_b=self.ref_texts[item], tgt_text=self.tgt_texts[item]
#                 # )
#             else:
#                 return f"<src_t> {self.src_trees[item]} <temp_t> {self.tgt_trees[item]}", f"<temp_t> {self.tgt_labels[item]}"
#             # return InputExample(
#             #     text_a=self.src_texts[item], text_b=self.tgt_texts[item], tgt_text=self.tgt_labels[item]
#             # )
#         else:
#             raise NotImplementedError

#     def __next__(self):
#         if self.current_iter == self.num_texts or self.trunc_size is not None and self.current_iter == self.trunc_size:
#             raise StopIteration

#         if self.has_ref:
#             res = InputExample(
#                 # guid=self.current_iter,
#                 text_a=self.src_trees[self.current_iter],
#                 text_b=self.ref_trees[self.current_iter],
#                 tgt_text=self.tgt_texts[self.current_iter]
#             )
#             self.current_iter += 1

#         else:
#             res = InputExample(
#                 # guid=self.current_iter,
#                 text_a=self.src_trees[self.current_iter],
#                 text_b=self.tgt_trees[self.current_iter],
#                 tgt_text=self.tgt_labels[self.current_iter]
#             )
#             self.current_iter += 1

#         return res

#     def __len__(self):
#         return self.num_texts
    
if __name__ == '__main__':
    from transformers import T5Tokenizer
    from torch.utils.data import DataLoader
    tok = T5Tokenizer.from_pretrained('./pretrained-models/syntax-t5-base')
    d = ParaNMTTripleDataset('./data/ParaNMT50m_triple/val', 't5', tok, 5,)
    dl = DataLoader(d, 1, collate_fn=d.collate_fn)
    for idx, line in enumerate(dl):
        print(f'{idx =:=^30}')
        print(tok.batch_decode(line.input_ids)[0])
        print(tok.batch_decode(line.labels)[0])
        if idx >= 100:
            break