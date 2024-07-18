try:
    from tree import MyTree
except ImportError:
    from .tree import MyTree
import re
from copy import deepcopy

def bracket_tree_to_slash(bracket_tree: MyTree):
    """convert a `MyTree` instance 
    to ROOT S NP DT This /DT /NP VP VBZ is /VBZ NP PRP me /PRP /NP /VP . . /. /S /ROOT"""
    res_str = f'{bracket_tree.label()} '
    for child in bracket_tree:
        if isinstance(child, str):
            res_str += f'{child} '
        else:
            res_str += bracket_tree_to_slash(child)
    
    res_str += f'/{bracket_tree.label()} '
    return res_str

def bracket_to_slash(bracket_tree: str):
    """convert a tree linearized by bracket representations 
    i.e. (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .))) 
    to ROOT S NP DT This /DT /NP VP VBZ is /VBZ NP PRP me /PRP /NP /VP . . /. /S /ROOT"""
    bracket_tree = re.sub(r'([()])', r' \1 ', bracket_tree)
    bracket_tree = re.sub(r' +', ' ', bracket_tree)
    elements = [each for each in bracket_tree.split(' ') if each]
    # print(elements)
    root_stack = []
    num_nodes_to_push = 0
    res_str = ''
    for element in elements:
        # print(f"processing `{element}`, root stack: {', '.join(root_stack)}")
        if element == '(':
            num_nodes_to_push += 1
        else:
            if element == ')':
                res_str += f'/{root_stack.pop(len(root_stack) - 1)} '
            else:
                if num_nodes_to_push > 0:
                    root_stack.append(element)
                    num_nodes_to_push -= 1
                res_str += f'{element} '
    
    return res_str

def slash_to_bracket(bracket_tree: str):
    """convert a tree linearized by slash representations 
    i.e. ROOT S NP DT This /DT /NP VP VBZ is /VBZ NP PRP me /PRP /NP /VP . . /. /S /ROOT
    to (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .)))"""
    elements = [each for each in bracket_tree.replace('(', ' ( ').replace(')', ' ) ').split(' ') if each]
    # print(elements)
    node_stack = []
    # num_nodes_to_push = 0
    for element in elements:
        # print(f"processing `{element}`, root stack: {', '.join(root_stack)}")
        if element.startswith('/') and element != '/':
            node_name = element[1:]
            for idx, node in enumerate(node_stack[::-1]):
                if node == node_name and idx != 0: # ignore the last node in the stack, since there is certain situation like (. .)
                    node_stack[len(node_stack) - idx - 1] = f'({node}'
                    break
            
            node_stack[-1] = node_stack[-1] + ')'
        
        else:
            node_stack.append(element)
    
    return ' '.join(node_stack)

# (ROOT (S (NP (PRP It)) (VP (VBZ 's) (NP (NP (DT a) (NN wonder)) (SBAR (S (NP (NP (DT neither)) (PP (IN of) (NP (PRP us)))) (VP (VBZ has) (VP (VBN been) (VP (VBN hurt)))))))) . (.)))

def bracket_tree_to_production(bracket_tree: MyTree):
    res = f'{bracket_tree.label()}: '
    results_to_append = []
    for each in bracket_tree:
        if isinstance(each, str):
            res += f'{each} '
        else:
            res += f'{each.label()} '
            results_to_append.append(bracket_tree_to_production(each))
    
    for each in results_to_append:
        res += f'<sep> {each} '
    
    return res
        

def bracket_to_production(bracket_tree: str): 
    """convert a tree linearized by bracket representations 
    i.e. (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .))) 
    to ROOT: S <sep> S: NP VP <sep> NP: DT <sep> DT: This <sep> VP: VBZ NP <sep> VBZ: is..."""
    t = MyTree.fromstring(bracket_tree)
    return bracket_tree_to_production(t)


def bracket_tree_to_path_(bracket_tree: MyTree, precedents: list = None):
    """convert a `MyTree` instance to a list of `path` nodes
    i.e. (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .)))
    to [ ROOT S NP DT This ] , [ ROOT S VP VNZ is ] , [ ROOT S VP NP PRP me ] , [ ROOT S VP . . ]"""
    if precedents is None: precedents = []  # avoid using a shared empty list for all function calls
    precedents.append(bracket_tree.label())
    all_results = []
    current_continuous_span = []
    is_continuous_span = False
    for each in bracket_tree:
        if isinstance(each, str):
            if not is_continuous_span:
                is_continuous_span = True
            current_continuous_span.append(each)
            # all_results.append(deepcopy(precedents + [each]))
        else:
            is_continuous_span = False
            if current_continuous_span:
                all_results.append(deepcopy(precedents + current_continuous_span))
                current_continuous_span = []
            all_results.extend(bracket_tree_to_path_(each, deepcopy(precedents)))
    
    if current_continuous_span:
        all_results.append(deepcopy(precedents + current_continuous_span))
    
    return all_results


def bracket_to_path(bracket_tree: str):
    """convert a `MyTree` instance to a `path` description
    i.e. (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .)))
    to ROOT S NP DT This <sep> ROOT S VP VNZ is <sep> ROOT S VP NP PRP me <sep> ROOT S VP . ."""
    paths = bracket_tree_to_path_(bracket_tree, [])

    return re.sub(r' +', ' ', ' <sep> '.join([' '.join(path) for path in paths]))

def bracket_tree_to_description(bracket_tree: MyTree):
    """convert a `MyTree` instance to a description format (bracket-seperated, a node label consists of 4 constituents, `node_label`, `height`, `parent`, `sibling_id`) like belows:
    like a sentence (node A at height h is parent B's i-th child)
    i.e. (ROOT (S (NP (DT This)) (VP (VBZ is) (NP (PRP me))) (. .)))
    to (ROOT-0-<None><0>)
    """

# def bracket_to_description(bracket_tree: str):
    
#     bracket_tree = MyTree.fromstring(bracket_tree)
#     stack = []    
    # res_seq = ''

"(ROOT (S (S (NP (CD 2) (JJR More) (NN information)) (VP (MD can) (VP (VB be) (VP (VBN found) (PP (IN on) (NP (NP (DT the) (NNP Agency) (POS 's)) (NN website))))))) (: :) (S (NN ema.europa.eu) :)) (VP (VB Find) (NP (NP (NN medicine/Human) (NN medicines/Referrals/Tredaptive)) (, ,) (NP (NN Pelzont)) (CC and) (NP (NN Trevaclyn))))) (. .)))"

if __name__ == '__main__':
    from svgling import draw_tree
    from tqdm import trange
    num_samples = 100_0000
    tr = trange(num_samples)
    PRINT = False
    with open('../ParaNMT50m_original/text_data/para-nmt-50m-batch-00_tree.txt', 'r') as f:
        idx = 0
        while idx <= num_samples:
            if PRINT:
                print('===' * 30)
                print(f'Data {idx}'.center(90, ' '))
                print('===' * 30)
            t_str = next(f).strip()
            t: MyTree = MyTree.fromstring(t_str) 
            t.get_height(1)
            t_str_no_redundant_space = t.tostr_without_indent()
            if PRINT:
                print(t_str_no_redundant_space)
                print('===' * 30)
            t = t.prune_downwards(5, 1)
            t_str_no_redundant_space_pruned = t.tostr_without_indent()
            # draw_tree(t).get_svg().saveas(f"../sample_trees/ParaNMT50m_original/{idx:03d}.svg")
            slash_tree_seq = bracket_to_slash(t_str)
            if PRINT:
                print(slash_tree_seq)
                print('===' * 30)
            pruned_slash_tree_seq = bracket_to_slash(t_str_no_redundant_space_pruned)
            # print(bracket_tree_to_production(t))
            # print('===' * 30)
            # print(bracket_to_path(t))
            # print('===' * 30)
            restored = slash_to_bracket(slash_tree_seq)
            if PRINT:
                print(restored)
            restored_pruned = slash_to_bracket(pruned_slash_tree_seq)
            if '\xa0' not in t_str:
                assert restored == t_str_no_redundant_space, f'restored\n{restored}\n!= original\n{t_str_no_redundant_space}'
                assert restored_pruned == t_str_no_redundant_space_pruned, f'restored_pruned\n{restored_pruned})\n!= original_pruned\n{t_str_no_redundant_space_pruned}'
            idx += 1
            tr.update()
