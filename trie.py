from functools import total_ordering
import numpy as np


NODE_LABELS_AMOUNT_MAX = 6
LOG_DIFF_THR = -39.14


def log_add(log_a, log_b):
    if log_a < log_b:
        diff = log_a - log_b
        if diff < LOG_DIFF_THR:
            res = log_b
        else:
            res = log_b + np.log1p(np.exp(diff))
    else:
        diff = log_b - log_a
        if diff < LOG_DIFF_THR:
            res = log_a
        else:
            res = log_a + np.log1p(np.exp(diff))
    return res


@total_ordering
class TrieNode(object):
    def __init__(self, idx):
        self.children = {}
        self.idx = idx
        self.n_label = 0
        self.label = []
        self.score = []
        self.max_score = 0.0
        self.id = id(self)

    def __gt__(self, other):
        return self.id > other.id

    def __lt__(self, other):
        return self.id < other.id

    def __ge__(self, other):
        return self.id >= other.id

    def __le__(self, other):
        return self.id <= other.id

    def __eq__(self, other):
        return self.id == other.id


class Trie(object):
    def __init__(self, idx, idx_to_sym):
        self.root_node = TrieNode(idx)
        self.nodes_count = 0
        self.idx_to_sym = idx_to_sym

    def get_root(self):
        return self.root_node

    def insert(self, idx_seq, label, score):
        node = self.root_node
        for idx in idx_seq:
            if idx not in node.children:
                node.children[idx] = TrieNode(idx)
                self.nodes_count += 1
            node = node.children[idx]
        if node.n_label < NODE_LABELS_AMOUNT_MAX:
            node.label.append(label)
            node.score.append(score)
            node.n_label += 1
        return node

    def search(self, idx_seq):
        node = self.root_node
        for idx in idx_seq:
            if idx not in node.children:
                return None
            node = node.children[idx]
        return node

    def smear_node(self, node: TrieNode, mode="MAX"):
        node.max_score = -float(np.inf)
        for i in range(node.n_label):
            node.max_score = log_add(node.max_score, node.score[i])

        for child in node.children.values():
            self.smear_node(child, mode=mode)
            if mode == "LOGADD":
                node.max_score = log_add(node.max_score, child.max_score)
            else:
                if child.max_score > node.max_score:
                    node.max_score = child.max_score

    def smear(self, mode="MAX"):
        self.smear_node(self.root_node, mode=mode)

    @property
    def nodes_amount(self):
        return self.nodes_count

