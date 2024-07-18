import os
# import sys; sys.path = [*(set(sys.path) | set(os.path.abspath('..')))]
from nltk.tree import Tree
from data.TreeUtils.linearizing_tree import bracket_to_slash
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from typing import Literal
import random as rd
import h5py
import multiprocessing
import re
from typing import *
from copy import deepcopy
from tqdm import tqdm
import queue
import pickle as pkl
import torch

def _mp_process(elements, func, *args):
    global arg_dict
    results = []
    current_n_completes = 0
    chunksize = max(len(elements) // 1000, 1)
    for idx, element in enumerate(elements):
        # print(f'processing.')
        results.append(func(element, *args))
        current_n_completes += 1
        if current_n_completes >= chunksize:
            arg_dict['res_queue'].put(current_n_completes)
            current_n_completes = 0
        # arg_dict['res_dict'][start_idx + idx] = res
     
    if current_n_completes:
        arg_dict['res_queue'].put(current_n_completes)
    arg_dict['res_queue'].put(None) 

    return results


# def complete_callback(result):
    # global arg_dict
    # print(f'complete!')
    # arg_dict['res_queue'].put(None)
    # result.get() # raise exceptions if necessary

def multiprocessing_process(
    iterable: Iterable[Any],
    func: Callable,
    *args,
    n_processes: int = None,
    # chunksize: int = None
):
    """
    mapping an iterable with a `func` using multiprocessing  
    Args:
        iterable: iterable 
    """
    def pool_init(_arg_dict):
        global arg_dict
        arg_dict = _arg_dict
    
    
    
    print(f'testing...')
    n_items = len(iterable)
    for i in range(min(10, n_items)):
        func(iterable[i], *args)
        
    with multiprocessing.Manager() as man:
        q = man.Queue()
        # dic = man.dict()
        print(f'initializing pool...')
        pool = multiprocessing.Pool(initializer=pool_init, initargs=({'res_queue': q},), processes=n_processes)
        n_processes = pool._processes
        print(f'Pool initialization complete, total {n_processes} worker processes')
        # print(hasattr(iterable, '__len__'))
        chunksize, _remainder = divmod(n_items, n_processes)
        iterable_shards = [
            iterable[
                chunk_idx * chunksize: 
                ((chunk_idx + 1) * chunksize if chunk_idx < n_processes - 1 else n_items)
            ] for chunk_idx in range(n_processes)
        ]
        print(f'lengths: {[len(each) for each in iterable_shards]}')
        print(f'applying...')
        apply_results = [pool.apply_async(_mp_process, (shard, func, *args)) for shard in iterable_shards]
        tqdm_complete = tqdm(total=n_items)
        n_completes = 0
        while n_completes < n_processes:
            try:
                complete_cnt = q.get(timeout=1)
                if complete_cnt is None:
                    n_completes += 1
                else:
                    tqdm_complete.update(complete_cnt)
            except queue.Empty:
                if all([each.ready() for each in apply_results]):
                    break
            
        total_results = []
        pbar = tqdm(apply_results, desc=f'extracting results..')
        for idx, result in enumerate(pbar):
            total_results.extend(result.get())
            pbar.set_description(f'extracting results ({idx + 1}/{len(apply_results)})..')
        
    
    return total_results




def tree_tostr(t: Tree):
    return re.sub(r'[\n ]+', ' ', str(t))

# def get_random_tree(node_format: Literal['<node>', 'ABC'], node_cnt_penalty: float = 0.99):
#     return _get_random_tree(0, node_cnt_penalty)

# def _get_random_tree(node_format: str, current_node_idx: int, current_probability: float, node_cnt_penalty: float):
#     if node_format == '<node>':
#         label = f'<node_{current_node_idx % 100}>'
#     elif node_format == 'ABC':
#         label = chr(ord('A') + current_node_idx % 26)
#     else:
#         raise NotImplementedError(f'node format `{node_format}` not recognized')

#     child_cnt = rd.choices([0, 1, 2, 3, 4, 5], [1, 1, 4, 5, 4, 1, 1])

def tree_to_details(tree: Tree, hs_format: str = "HS"):
    """
    convert a `Tree` instance to a description format (bracket-seperated, a node label consists of 4 constituents, `node_label`, `height`, `parent`, `sibling_id`) like belows:
    ```
    (S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN mouse))))
    ```
    to
    ```
    (S<H_1><S_1><none> (NP<H_2><S_1>S (DT<H_3><S_1>NP the) (NN<H_3><S_2>NP cat)) (VP<H_2><S_2>S (VBD<H_3><S_1>VP ate) (NP<H_3><S_2>VP (DT<H_4><S_1>NP a) (NN<H_4><S_2>NP mouse))))
    """
    for each in tree.treepositions():
        # if isinstance(tree[each], str):
        #     tree[each] = Tree(tree[each])
        height = len(each) + 1
        p_label = tree[each[:-1]].label() if len(each) else None
        sibling_idx = each[-1] + 1 if len(each) else None
        if len(hs_format) == 2:
            if hs_format != '--':
                new_label = f'{tree[each] if isinstance(tree[each], str) else tree[each].label()}<{hs_format[0]}_{height}><{hs_format[1]}_{sibling_idx or 1}>{p_label or "<none>"}'
            else:
                new_label = f'{tree[each] if isinstance(tree[each], str) else tree[each].label()}-{height}-{sibling_idx or 1}-{p_label or "<none>"}'
        else:
            raise AttributeError(f'argument `hs_format` must be a string of length 2')
        # if isinstance(tree[each], str):
        #     tree_each = new_label
        if not isinstance(tree[each], str):
            tree[each]._new_label = new_label
        
    for each in tree.treepositions():
        if not isinstance(tree[each], str):
            tree[each].set_label(tree[each]._new_label)
            del tree[each]._new_label
    
    return tree_tostr(tree)


def bracket_to_details(bracket_line: str, hs_format: str = 'HS'):
    # line_split = [*filter(lambda x: x, bracket_line.replace('(', ' ( ').replace(')', ' ) ').split())]
    tree: Tree = Tree.fromstring(bracket_line)
    return tree_to_details(tree, hs_format)


def get_tokenizer_and_model_with_node_tokens(pretrained_model_name_or_path: str, export_path: str):
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    # add tokens
    tokenizer.add_tokens([f'<node_{i}>' for i in range(100)])
    tokenizer.add_tokens([f'</node_{i}>' for i in range(100)])
    tokenizer.add_tokens([f'<H_{i}>' for i in range(1, 41)])
    tokenizer.add_tokens([f'<S_{i}>' for i in range(1, 41)])
    tokenizer.add_tokens([f'<none>'])
    tokenizer.add_tokens([f'<sep>'])
    # resize model embeddings and initialize embeddings
    prev_embedding_size = model.get_input_embeddings().weight.data.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.get_input_embeddings()
    # model.set_input_embeddings()
    def copy_token_embedding(to_token: str, from_token: str):
        from_token_idx, to_token_idx = tokenizer(from_token, add_special_tokens=False).input_ids, tokenizer(to_token, add_special_tokens=False).input_ids
        assert len(to_token_idx) == 1, f'error, `from_token` or `to_token` split into subtokens: (lengths: from_token_idx: {from_token_idx}, to_token_idx: {to_token_idx})'
        embedding.weight.data[to_token_idx[0]] = embedding.weight.data[from_token_idx[-1]] # if `to_token`'s tokenized length > 1, we assume that the first token is white space char (U+2581) 

    for i in range(100):
        copy_token_embedding(f'<node_{i}>', f'{i}')
        copy_token_embedding(f'</node_{i}>', f'{i}')
    for i in range(1, 41):
        copy_token_embedding(f'<H_{i}>', f'{i}')
        copy_token_embedding(f'<S_{i}>', f'{i}')
    copy_token_embedding('<none>', '<unk>')
    copy_token_embedding('<sep>', '<unk>')
    # model.set_input_embeddings(embedding)

    new_embedding_size = embedding.weight.data.shape[0]
    print(f'embedding expanded {new_embedding_size - prev_embedding_size} tokens from {prev_embedding_size} to {new_embedding_size} (vocabulary size: {len(tokenizer)})')
    tokenizer.save_pretrained(export_path)
    model.save_pretrained(export_path)

def add_NT_tokens_to_model(pretrained_model_name_or_path: str, pretrained_tokenizer_name_or_path: str, export_path: str, initialize: Literal['node', 'unk'] = 'node'):
    """
    add syntax tree non-terminal tokens to model
    """
    # load model and tokenizer
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(pretrained_tokenizer_name_or_path)
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
    NT_tokens = pkl.load(open('../data/ParaNMT50m_original/NT.pkl', 'rb'))
    tokenizer.add_tokens(NT_tokens)
    # resize model embeddings and initialize embeddings by sampling with mean=node_embedding's element-wise mean, var=node_embedding's element-wise variance
    model.resize_token_embeddings(len(tokenizer))
    embedding = model.get_input_embeddings()
    if initialize == 'node':
        node_embeddings = embedding.weight.data[[tokenizer(f'<node_{i}>', add_special_tokens=False).input_ids[0] for i in range(100)]]
        node_emb_mean, node_emb_std = node_embeddings.mean(dim=0), node_embeddings.std(dim=0)
        for nt_token in NT_tokens:
            embedding.weight.data[tokenizer(nt_token, add_special_tokens=False).input_ids[0]] = torch.randn(node_emb_mean.shape) * node_emb_std + node_emb_mean
    elif initialize == 'unk':
        unk_embedding = embedding.weight.data[tokenizer('<unk>', add_special_tokens=False).input_ids[0]]
        for nt_token in NT_tokens:
            embedding.weight.data[tokenizer(nt_token, add_special_tokens=False).input_ids[0]] = unk_embedding

    tokenizer.save_pretrained(export_path)
    model.save_pretrained(export_path)


def convert_constituency_to_random_tree(
    # args: List[Any]
    constituency_seq: str, 
    node_format: Literal['<node>', 'ABC'], 
    tree_format: Literal['bracket', 'slash', 'detail']
    ):
    # idx, constituency_seq, node_format, tree_format = args
    # print(f'processing [{idx}] (`{constituency_seq}`)')
    if isinstance(constituency_seq, bytes):
        constituency_seq = constituency_seq.decode()
    tree: Tree = Tree.fromstring(constituency_seq)
    if node_format == '<node>':
        labels = [f'<node_{i}>' for i in range(100)]
    elif node_format == 'ABC':
        labels = [chr(ord('A') + i) for i in range(26)]
    else:
        raise NotImplementedError(f'node format `{node_format}` not recognized')
    
    rd.shuffle(labels)

    for pos_idx, pos in enumerate(tree.treepositions()):
        current_label = labels[pos_idx % len(labels)]
        if isinstance(tree[pos], str):
            tree[pos[:-1]][pos[-1]] = current_label
        else:
            tree[pos].set_label(current_label)
    
    # print('complete')
    return tree_tostr(tree) if tree_format == 'bracket' else bracket_to_slash(tree_tostr(tree)) # re.sub(r' +', ' ', str(tree).replace('\n', ''))

def convert_all_constituency_trees_to_random_trees(
        constituency_file_path: str,
        node_format: Literal['<node>', 'ABC'], 
        tree_format: Literal['bracket', 'slash', 'detail'],
        truncation_cnt: int = None,
    ):
    with h5py.File(constituency_file_path, 'r') as f:
        # for each in zip(range(10_000), tqdm(f['trees'])):
        #     pass
        # print(*[convert_constituency_to_random_tree(idx, each, node_format, tree_format) for idx, each in enumerate(f['trees'][:10])], sep='\n')
        # num_trees = len(f['trees'])
        if truncation_cnt is not None:
            results = multiprocessing_process(f['trees'][:truncation_cnt], convert_constituency_to_random_tree, node_format, tree_format)
        else:
            results = multiprocessing_process(f['trees'], convert_constituency_to_random_tree, node_format, tree_format)

    file_name = f'random_trees_{node_format}_{tree_format}'
    if truncation_cnt is not None:
        file_name += f'_{truncation_cnt}'
    file_name += '.h5'
    with h5py.File(os.path.join(os.path.split(constituency_file_path)[0], file_name), 'w') as f:
        f.create_dataset('trees', data=results)


def format_production(production: nltk.grammar.Production):
    return f'{production.lhs().symbol()} -> ' + ' '.join(map(lambda x: x if isinstance(x, str) else x.symbol(), production.rhs()))

def get_nonterminal_treepositions(tree: Tree):
    terminal_positions = [tree.leaf_treeposition(each) for each in range(len(tree.leaves()))]
    return [each for each in tree.treepositions() if each not in terminal_positions]

def delete_node(tree: Tree):
    _tree = deepcopy(tree)
    treepositions = [each for each in get_nonterminal_treepositions(_tree) if len(each) >= 1]
    max_length = max([len(x) for x in treepositions])
    # treeposition = rd.choices(tree, map(lambda x: max_length - len(x) + 1, treepositions)) 
    treeposition = rd.choice(treepositions)
    children = [*_tree[treeposition]]
    parent = _tree[treeposition[:-1]]
    # try:
    parent.pop(treeposition[-1]) # remove selected node
    # except IndexError:
    #     print(f'indexerror occurred!!!')
    #     print(_tree[treeposition], treeposition)
    for idx, each in enumerate(children):
        # print(f'inserting child {each} at position {treeposition[-1] + idx}')
        parent.insert(treeposition[-1] + idx, each)
    
    return treeposition, _tree


class TreeTaskFormatter:
    @staticmethod
    def line_to_treeposition_indexing(line: str, linearize_format: Literal['bracket', 'slash']):
        tree: Tree = Tree.fromstring(line)
        treepositions = tree.treepositions()
        treeposition = rd.choices(treepositions, [*map(lambda x: len(x), treepositions)])[0] 
        if linearize_format == 'slash':
            line = bracket_to_slash(line)
        # truncate_idx = rd.choices([*range(len(treeposition)), range(1, len(treeposition) + 1)]) 
        # treeposition = treeposition[:truncate_idx]
        return f"treeposition indexing position: {' '.join(map(lambda x: str(x + 1), treeposition))} tree: {line}", tree[treeposition] if isinstance(tree[treeposition], str) else tree[treeposition].label()
    
    @staticmethod
    def line_to_tree_forming(line: str, linearize_format: Literal['bracket', 'slash']):
        tree: Tree = Tree.fromstring(line)
        productions = tree.productions()
        if linearize_format == 'slash':
            line = bracket_to_slash(line)
        return f"tree forming: {' <sep> '.join(map(format_production, productions))}", line 
        
    
    @staticmethod
    def line_to_node_deleting(line: str, linearize_format: Literal['bracket', 'slash']):
        tree: Tree = Tree.fromstring(line)
        treeposition, deleted_tree = delete_node(tree)
        deleted_tree = tree_tostr(deleted_tree)
        if linearize_format == 'slash':
            deleted_tree = bracket_to_slash(deleted_tree)
            line = bracket_to_slash(line)
        # print(treeposition) 
        # print(line) 
        # print(deleted_tree)
        return f"node deleting position: {', '.join(map(lambda x: str(x + 1), treeposition))} node: {tree[treeposition].label()} tree: {line}", deleted_tree
    
    @staticmethod
    def line_to_height_selection(line: str, linearize_format: Literal['bracket', 'slash']):
        tree: Tree = Tree.fromstring(line)
        treepositions = tree.treepositions()
        lengths = [len(each) for each in treepositions]
        tgt_length = rd.choice(lengths)
        height_treepositions = [each for each in treepositions if len(each) == tgt_length]
        height_node_labels = [(tree[each] if isinstance(tree[each], str) else tree[each].label()) for each in height_treepositions ]
        if linearize_format == 'slash':
            line = bracket_to_slash(line)
        return f"height selection height: {tgt_length} tree: {line}", " ".join(height_node_labels)
    
    @staticmethod
    def line_to_tree_interpreting(line: str, linearize_format: Literal['bracket', 'slash']):
        converted_tree = bracket_to_details(line)
        if linearize_format == 'slash':
            line = bracket_to_slash(line)
            converted_tree = bracket_to_slash(converted_tree)
        
        return f"tree interpreting: {line}", converted_tree


    tasks = ['treeposition_indexing', 'tree_forming', 'node_deleting', 'height_selection', 'tree_interpreting']
    linearizing_formats = ['bracket', 'slash']

    def get_method(self, task: str, linearize_format: Literal['bracket', 'slash']) -> Callable[[str], Tuple[str, str]]:
        assert linearize_format in ['bracket', 'slash'], f'error, linearizing format `{linearize_format}` not recognized'
        return lambda line: getattr(self, f"line_to_{task}")(line, linearize_format)

def test_formatting_method(tree_file: str, formatting_method: str, n_trees: int = 10):
    tf = TreeTaskFormatter()
    tok = T5Tokenizer.from_pretrained('../pretrained-models/syntax-t5-base-node')
    with h5py.File(tree_file, 'r') as f:
        for each in f['trees'][:n_trees]:
            print('===' * 30)
            inputs, labels = tf.get_method(formatting_method, 'bracket')(each.decode())
            print(f'inputs: {inputs}')
            print(f'length: {len(tok.tokenize(inputs))}')
            print(f'labels: {labels}')
            print(f'length: {len(tok.tokenize(labels))}')

if __name__ == '__main__':
    # test_formatting_method('../data/ParaNMT50m_original/random_trees_<node>_bracket_1000000.h5', 'tree_interpreting')
    # convert_all_constituency_trees_to_random_trees('../data/ParaNMT50m_original/train_data.h5', '<node>', 'bracket', 100_0000)
    # convert_all_constituency_trees_to_random_trees('../data/ParaNMT50m_original/train_data.h5', '<node>', 'bracket')
    # convert_all_constituency_trees_to_random_trees('../data/ParaNMT50m_original/train_data.h5', '<node>', 'slash')
    # with h5py.File('../data/ParaNMT50m_original/train_data.h5', 'r') as f:
    #     tf = TreeTaskFormatter()
    #     for idx, tree in zip(range(10), f['trees']):
    #         print('==' * 30)
    #         print(*tf.line_to_node_deleting(tree.decode(), 'bracket'), sep='\n' + '=' * 30 + '\n')
    # get_tokenizer_and_model_with_node_tokens('../pretrained-models/t5-base', '../pretrained-models/syntax-t5-base-node')
    # add_NT_tokens_to_model('./runs/tree_seed_test/113/checkpoints/model/',  '../pretrained-models/syntax-t5-base-node',  '../pretrained-models/syntax-t5-base-node-with-NT')
    add_NT_tokens_to_model('../pretrained-models/syntax-t5-base-node',  '../pretrained-models/syntax-t5-base-node',  '../pretrained-models/syntax-t5-base-node-unk-NT', 'unk')
    # import time
    # prev_time = time.time()
    # print(bracket_to_details('(S (NP (DT the) (NN cat)) (VP (VBD ate) (NP (DT a) (NN mouse))))'))
    # print(f'{time.time() - prev_time}s elapsed')
