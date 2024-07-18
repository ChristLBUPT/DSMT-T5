from svgling import draw_tree
from time import time
from nltk.tree import Tree

# import stanza
# from zss import Node as ZSSNode
# from stanza.models.constituency.parse_tree import Tree as StanzaTree
# from stanza.protobuf.CoreNLP_pb2 import ParseTree
# import json

import re
from typing import *


class MyTree(Tree):
    class RetainChildren:
        """During pruning, assume the prune height is `h`, if a node at level `h`
        still has children, all leaf nodes in its children might be "moved up" to become the node's
        direct children
        """
        NONE = 0  # do not retain children (original prune logic)
        ORIGINAL = 1  #
        MASK = 2

    def __init__(self, node: str, children: List[Any] = None):
        super(MyTree, self).__init__(node=node, children=children)
        self.my_height = -1
        self.my_depth = -1
        # self.label = self._label
        # self.children = self

    def get_height(self, current_height: int):
        self.my_height = current_height
        for child in self:
            if not isinstance(child, str):
                child.get_height(current_height + 1)
    

    def get_depth(self):
        depths = []
        for child in self:
            if isinstance(child, str):
                depths.append(1)
            else:
                depths.append(child.get_depth())
        self.my_depth = max(depths)

    # children
    @staticmethod
    def from_other(other_node):
        if hasattr(other_node, 'label') and hasattr(other_node, 'children'):
            other_name, other_children = other_node.label, other_node.children
        elif hasattr(other_node, 'value') and hasattr(other_node, 'child'):
            other_name, other_children = other_node.value, other_node.child
        else:
            raise AssertionError('error, other node has neither `label` and `children` nor `value` and `child`')
        if not other_children:
            return other_name
        else:
            return MyTree(other_name, [MyTree.from_other(each) for each in other_children])

    def to_other(self, other_type):
        return other_type(self._label,
                          [other_type(each, None) if isinstance(each, str) else each.to_other(other_type) for each in
                           self])

    def get_all_leaves(self):
        res = []
        for child in self:
            if isinstance(child, str):
                res.extend([child])
            else:
                res.extend(child.get_all_leaves())

        return res

    def prune_downwards(self, prune_height: int, retain_children: int = 0, inference_height: bool = False):
        """
        if there is a tree (S (NP I) (VP (VBD am) (NN king))) to be pruned at height 2, then
        0(None): (S NP VP)
        1(Original/Retain Children): (S (NP I) (VP am king))
        2(Mask): (S (NP <mask>) (VP <mask>))
        """
        new_tree = MyTree(self.label(), [])
        if not inference_height:
            assert self.my_height != -1, "Error: must use `get_height()` to calculate height first! (pass `inference_height=True` to supress this error)"
        else:
            if self.my_height == -1:
                self.get_height(1)
        if self.my_height + 1 <= prune_height:
            for child in self:
                if isinstance(child, str):
                    if retain_children == MyTree.RetainChildren.ORIGINAL:
                        new_tree.append(child)
                    elif retain_children == MyTree.RetainChildren.MASK:
                        new_tree.append('<mask>')
                    # if retain_children == MyTree.RetainChildren.ORIGINAL:
                    #     new_tree.append(child)
                    # elif retain_children == MyTree.RetainChildren.MASK:
                    #     new_tree.append('<mask>')
                else:
                    new_tree.append(child.prune_downwards(prune_height, retain_children))

            return new_tree

        if self.my_height == prune_height:
            if retain_children == MyTree.RetainChildren.ORIGINAL:
                return MyTree(self.label(), [' '.join(self.get_all_leaves())])
            elif retain_children == MyTree.RetainChildren.MASK:
                return MyTree(self.label(), ['<mask>'])
            else:
                return self.label()

    def tostr_without_indent(self):
        return re.sub(r'[\n ]+', ' ', str(self))

    def tostr_with_bracket_spaces(self):
        return re.sub(
            r' +', ' ',
                re.sub(
                r'([()])', r' \1 ',
                re.sub(r'[\n ]+', ' ', str(self))
            )
        )


    def replace_leaf_with_mask(self):
        nt = MyTree(self.label(), [])
        for child in self:
            if isinstance(child, str):
                nt.append('<mask>')
            else:
                nt.append(child.replace_leaf_with_mask())

        return nt

    def remove_all_leaves(self):
        return_str = True
        for child in self:
            if not isinstance(child, str):
                return_str = False

        if return_str:
            return self.label()
        else:
            nt = MyTree(self.label(), [])
            for child in self:
                if not isinstance(child, str):
                    nt.append(child.remove_all_leaves())

            return nt


if __name__ == '__main__':
    t1 = time()
    # mt = MyTree('A', [MyTree('B', ['e', MyTree('F', ['g', 'h'])]), 'c', 'd'])
    mt = MyTree.fromstring('(ROOT (FRAG (SBAR (WHADJP (WRB how) (JJ exciting)) (S (NP (DT that)) (VP (VBZ is)))) (. .)))')
    mt.get_height(1)
    draw_tree(mt).get_svg().saveas('original.svg')
    draw_tree(mt.prune_downwards(2)).get_svg().saveas('prune2.svg')
    draw_tree(mt.remove_all_leaves()).get_svg().saveas('without_children.svg')
    t2 = time()
    print(f'costed {t2 - t1:.6f}s')
    # pip = stanza.Pipeline(processors='tokenize,pos,constituency', download_method=None)
    # res = pip.process('how exciting that is . that is exactly right . that is really exciting .')
    # src_parse = MyTree.from_other(res.sentences[0].constituency)
    # ref_parse = MyTree.from_other(res.sentences[1].constituency)
    # tgt_parse = MyTree.from_other(res.sentences[2].constituency)
    # svgling.draw_tree(src_parse).get_svg().saveas('src.svg')
    # svgling.draw_tree(ref_parse).get_svg().saveas('ref.svg')
    # svgling.draw_tree(tgt_parse).get_svg().saveas('tgt.svg')
    # fully_linearize = lambda src: re.sub(r'[\n ]+', ' ', src)
    # print(src_parse[0][0][0].get_all_leaves())
    # print('src:', fully_linearize(str(src_parse)))
    # print('ref:', fully_linearize(str(ref_parse)))
    # print('tgt:', fully_linearize(str(tgt_parse)))
    #
    # test_tree = MyTree('a', [MyTree('b', ['c', MyTree('d', ['e'])]), 'f', MyTree('g', ['h'])])  # MyTree('h', [])
    # print(test_tree)
    # # print(test_tree.children)
    # test_tree.pretty_print()
    # zss: StanzaTree = test_tree.to_other(StanzaTree)
    # print(zss.pretty_print())
