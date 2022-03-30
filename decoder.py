import time
import kenlm
import numpy as np
from operator import attrgetter
import os

import trie


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MINUS_INF = -float(np.inf)
SPACE_WEIGHT = 1.0
WORD_SCORE = 0.0
LM_WEIGHT = 1.0
UNK_SCORE = MINUS_INF
BEAM_SCORE = 10.
BEAM_SIZE = 1000
LOG_ADD = False


def compare_lm_states(lm_state_1: kenlm.State, lm_state_2: kenlm.State):
    return lm_state_1.__eq__(lm_state_2)


class LexiconDecoderNode(object):
    def __init__(self, lm_state, lex_node, parent, score, amscore, label=None, blank=False):
        self.lm_state = lm_state
        self.lex_node = lex_node
        self.parent = parent
        self.score = score
        self.amscore = amscore
        self.label = label
        self.blank = blank


class LexiconDecoder(object):
    def __init__(self, lexicon: trie.Trie, lm: kenlm.Model, sil_idx: int, blank_idx: int, space_idx: int, idx_to_word):
        self.lexicon = lexicon
        self.lm = lm
        self.candidates = []
        self.sil_idx = sil_idx
        self.space_idx = space_idx
        self.blank_idx = blank_idx
        self.idx_to_word = idx_to_word
        self.candidates_best_score = MINUS_INF
        self.hyps = None

    def reset_candidates(self):
        self.candidates = []
        self.candidates_best_score = MINUS_INF

    def merge_nodes(self, node_1: LexiconDecoderNode, node_2: LexiconDecoderNode, log_add=LOG_ADD):
        max_score = max(node_1.score, node_2.score)
        max_amscore = max(node_1.amscore, node_2.amscore)
        if log_add:
            node_1.score = max_score + np.log(np.exp(node_1.score - max_score) + np.exp(node_2.score - max_score))
            node_1.amscore = max_amscore + np.log(np.exp(node_1.amscore - max_amscore) + np.exp(node_2.amscore - max_amscore))
        else:
            node_1.score = max_score
            node_1.amscore = max_amscore
        return node_1

    def add_candidate(self, lm_state, lex, parent, score, amscore, label, prev_blank):
        if score > self.candidates_best_score:
            self.candidates_best_score = score
        if score >= self.candidates_best_score - BEAM_SCORE:
            self.candidates.append(LexiconDecoderNode(lm_state, lex, parent, score, amscore, label, prev_blank))

    def store_candidates(self, t):
        """ select valid candidates """
        valid_candidates = []
        for i in range(len(self.candidates)):
            if self.candidates[i].score >= self.candidates_best_score - BEAM_SCORE:
                valid_candidates.append(self.candidates[i])

        """ sort by (lm_state, lexicon, score) and copy into next hyps """
        valid_candidates = sorted(set(valid_candidates), key=attrgetter("lm_state", "lex_node", "score"), reverse=True)

        idx = 0
        for i in range(1, len(valid_candidates)):
            if ((compare_lm_states(valid_candidates[idx].lm_state, valid_candidates[i].lm_state)) &
                    (valid_candidates[idx].lex_node == valid_candidates[i].lex_node)):
                valid_candidates[idx] = self.merge_nodes(valid_candidates[idx], valid_candidates[i])
            else:
                idx += 1
                valid_candidates[idx] = valid_candidates[i]

        valid_candidates = valid_candidates[: idx + 1]

        """ sort candidates by score and select BEAM SIZE """
        valid_candidates = sorted(set(valid_candidates), key=attrgetter("score"), reverse=True)
        self.hyps[t+1] = valid_candidates[: BEAM_SIZE]

    def decode(self, m, id=None):
        T = m.shape[0]
        N = m.shape[1]        

        self.hyps = np.empty(T + 3, dtype=np.object)

        lm_init_state = kenlm.State()
        self.lm.BeginSentenceWrite(lm_init_state)
        lex_init_node = self.lexicon.get_root()

        self.hyps[0] = []
        self.hyps[0].append(LexiconDecoderNode(lm_init_state, lex_init_node, None, 0.0, 0.0, None))

        add_candidates_time = 0
        select_candidates_time = 0

        for t in range(T):
            self.reset_candidates()

            start_t_time = time.time()

            for k in range(len(self.hyps[t])):
                prev_hyp_idx = k
                h = self.hyps[t][k]
                prev_lex_node = h.lex_node
                prev_lex_idx = prev_lex_node.idx
                lex_max_score = prev_lex_node.max_score
                prev_lm_state = h.lm_state
                prev_blank = h.blank

                for n in range(N):
                    score = h.score + m[t, n]
                    amscore = h.amscore + m[t, n]

                    """ emit a word only if space """
                    if n == self.space_idx:
                        score = score + SPACE_WEIGHT

                        """ if we got a true word """
                        for i in range(prev_lex_node.n_label):
                            new_lm_state = kenlm.State()
                            lm_score = self.lm.BaseScore(prev_lm_state,
                                                         self.idx_to_word[prev_lex_node.label[i].lm_idx],
                                                         new_lm_state)

                            score = score + LM_WEIGHT * (lm_score - lex_max_score) + WORD_SCORE
                            self.add_candidate(new_lm_state, self.lexicon.get_root(), prev_hyp_idx, score, amscore,
                                               prev_lex_node.label[i], False)

                        """ if we got unknown word """
                        if (prev_lex_node.n_label == 0) & (UNK_SCORE > MINUS_INF):
                            pass

                        """ if we got space char again """
                        if (n == prev_lex_idx) & (not prev_blank):
                            self.add_candidate(prev_lm_state, prev_lex_node, prev_hyp_idx, score, amscore, None, False)

                        """ allow starting with a space """
                        if t == 0:
                            self.add_candidate(prev_lm_state, prev_lex_node, prev_hyp_idx, score, amscore, None, False)
                    elif (n == prev_lex_idx) & (not prev_blank):
                        self.add_candidate(prev_lm_state, prev_lex_node, prev_hyp_idx, score, amscore, None, False)
                    elif n == self.blank_idx:
                        self.add_candidate(prev_lm_state, prev_lex_node, prev_hyp_idx, score, amscore, None, True)
                    else:
                        if n in prev_lex_node.children:
                            lex_node = prev_lex_node.children[n]
                            score = score + LM_WEIGHT * (lex_node.max_score - lex_max_score)
                            self.add_candidate(prev_lm_state, lex_node, prev_hyp_idx, score, amscore, None, False)

            end_t_time = time.time()

            add_candidates_time += end_t_time - start_t_time

            self.store_candidates(t)

            select_candidates_time += time.time() - end_t_time

        """ finish up """
        self.reset_candidates()
        for k in range(len(self.hyps[T])):
            prev_hyp_idx = k
            h = self.hyps[T][k]
            prev_lex_node = h.lex_node
            prev_lex_idx = prev_lex_node.idx
            lex_max_score = prev_lex_node.max_score
            prev_lm_state = h.lm_state

            for i in range(prev_lex_node.n_label):
                new_lm_state = kenlm.State()
                lm_score = self.lm.BaseScore(prev_lm_state,
                                             self.idx_to_word[prev_lex_node.label[i].lm_idx],
                                             new_lm_state)
                lm_score_end = self.lm.BaseScore(new_lm_state, "</s>", new_lm_state)
                score = h.score + LM_WEIGHT * (lm_score + lm_score_end - lex_max_score) + WORD_SCORE
                self.add_candidate(new_lm_state, self.lexicon.get_root(), prev_hyp_idx, score, amscore, prev_lex_node.label[i], False)

            if prev_lex_idx == self.space_idx:
                new_lm_state = kenlm.State()
                lm_score_end = self.lm.BaseScore(prev_lm_state, "</s>", new_lm_state)
                score = h.score + LM_WEIGHT * lm_score_end
                self.add_candidate(new_lm_state, prev_lex_node, prev_hyp_idx, score, amscore, None, False)

        self.store_candidates(T)

        return self.store_hyps(T)

    def store_hyps(self, T):
        hyps = []
        n_hyp = len(self.hyps[T+1])
        prev_idx = self.sil_idx
        for i in range(n_hyp):
            hyp = {}
            node = self.hyps[T + 1][i]
            hyp['score'] = node.score
            hyp['amscore'] = node.amscore
            word_timestamps = []
            word_hyp = []
            t = T + 1
            is_end = True
            end_timestamp = T
            while t > 0:
                if is_end:
                    if node.lex_node.idx == self.sil_idx or node.lex_node.idx == self.space_idx:
                        end_timestamp = t - 1
                    else:
                        print()
                        is_end = False

                if (node.lex_node.idx == self.sil_idx or node.lex_node.idx == self.space_idx) \
                    and prev_idx != self.sil_idx and prev_idx != self.space_idx:
                    word_timestamps.append(t - 1)
                
                if node.label is not None:
                    word_hyp.append(node.label.lm_idx)

                prev_idx = node.lex_node.idx
                t -= 1
                node = self.hyps[t][node.parent]
            word_hyp.reverse()
            word_timestamps.reverse()
            hyp['word_timestamps'] = word_timestamps
            hyp['word_hyp'] = word_hyp
            hyp['end_timestamp'] = end_timestamp
            hyps.append(hyp)
        return hyps


class LexiconFreeDecoderNode(object):
    def __init__(self, idx, lm_state, parent, score, amscore, blank=False):
        self.idx = idx
        self.lm_state = lm_state
        self.parent = parent
        self.score = score
        self.amscore = amscore
        self.blank = blank


class LexiconFreeDecoder(object):
    def __init__(self, lm: kenlm.Model, sil_idx: int, blank_idx: int, space_idx: int, idx_to_sym):
        self.lm = lm
        self.candidates = []
        self.sil_idx = sil_idx
        self.space_idx = space_idx
        self.blank_idx = blank_idx
        self.idx_to_sym = idx_to_sym
        self.candidates_best_score = MINUS_INF
        self.hyps = None

    def reset_candidates(self):
        self.candidates = []
        self.candidates_best_score = MINUS_INF

    def merge_nodes(self, node_1: LexiconFreeDecoderNode, node_2: LexiconFreeDecoderNode, log_add=LOG_ADD):
        max_score = max(node_1.score, node_2.score)
        max_amscore = max(node_1.amscore, node_2.amscore)
        if log_add:
            node_1.score = max_score + np.log(np.exp(node_1.score - max_score) + np.exp(node_2.score - max_score))
            node_1.amscore = max_amscore + np.log(np.exp(node_1.amscore - max_amscore) + np.exp(node_2.amscore - max_amscore))
        else:
            node_1.score = max_score
            node_1.amscore = max_amscore
        return node_1

    def add_candidate(self, idx, lm_state, parent, score, amscore, prev_blank):
        if score > self.candidates_best_score:
            self.candidates_best_score = score
        if score >= self.candidates_best_score - BEAM_SCORE:
            self.candidates.append(LexiconFreeDecoderNode(idx, lm_state, parent, score, amscore, prev_blank))

    def store_candidates(self, t):
        """ select valid candidates """
        valid_candidates = []
        for i in range(len(self.candidates)):
            if self.candidates[i].score >= self.candidates_best_score - BEAM_SCORE:
                valid_candidates.append(self.candidates[i])

        """ sort by (lm_state, lexicon, score) and copy into next hyps """
        valid_candidates = sorted(set(valid_candidates), key=attrgetter("lm_state", "idx", "score"), reverse=True)

        idx = 0
        for i in range(1, len(valid_candidates)):
            if ((compare_lm_states(valid_candidates[idx].lm_state, valid_candidates[i].lm_state)) &
                    (valid_candidates[idx].idx == valid_candidates[i].idx)):
                valid_candidates[idx] = self.merge_nodes(valid_candidates[idx], valid_candidates[i])
            else:
                idx += 1
                valid_candidates[idx] = valid_candidates[i]

        valid_candidates = valid_candidates[: idx + 1]

        """ sort candidates by score and select BEAM SIZE """
        valid_candidates = sorted(set(valid_candidates), key=attrgetter("score"), reverse=True)
        self.hyps[t+1] = valid_candidates[: BEAM_SIZE]

    def decode(self, m, id=None):

        T = m.shape[0]
        N = m.shape[1]        

        self.hyps = np.empty(T + 3, dtype=np.object)

        lm_init_state = kenlm.State()
        self.lm.BeginSentenceWrite(lm_init_state)
        init_idx = self.space_idx

        self.hyps[0] = []
        self.hyps[0].append(LexiconFreeDecoderNode(init_idx, lm_init_state, None, 0.0, 0.0, False))

        add_candidates_time = 0
        select_candidates_time = 0

        for t in range(T):
            self.reset_candidates()

            start_t_time = time.time()

            for k in range(len(self.hyps[t])):
                prev_hyp_idx = k
                h = self.hyps[t][k]
                prev_idx = h.idx
                prev_lm_state = h.lm_state
                prev_blank = h.blank

                for n in range(N):
                    score = h.score + m[t, n]
                    amscore = h.amscore + m[t, n]

                    if n != self.blank_idx and (n != prev_idx or prev_blank):
                        new_lm_state = kenlm.State()
                        lm_score = self.lm.BaseScore(prev_lm_state, self.idx_to_sym[n], new_lm_state)
                        score = score + LM_WEIGHT * lm_score
                        self.add_candidate(n, new_lm_state, prev_hyp_idx, score, amscore, False)
                    elif n == self.blank_idx:
                        self.add_candidate(prev_idx, prev_lm_state, prev_hyp_idx, score, amscore, True)
                    else:
                        self.add_candidate(prev_idx, prev_lm_state, prev_hyp_idx, score, amscore, False)

            end_t_time = time.time()

            add_candidates_time += end_t_time - start_t_time

            self.store_candidates(t)

            select_candidates_time += time.time() - end_t_time

        """ finish up """
        self.reset_candidates()
        for k in range(len(self.hyps[T])):
            prev_hyp_idx = k
            h = self.hyps[T][k]
            prev_idx = h.idx
            prev_lm_state = h.lm_state
            prev_blank = h.blank

            new_lm_state = kenlm.State()
            lm_score_end = self.lm.BaseScore(prev_lm_state, "</s>", new_lm_state)
            score = h.score + LM_WEIGHT * lm_score_end
            self.add_candidate(prev_idx, new_lm_state, prev_hyp_idx, score, amscore, False)

        self.store_candidates(T)

        return self.store_hyps(T)

    def store_hyps(self, T):
        hyps = []
        n_hyp = len(self.hyps[T+1])
        prev_idx = self.sil_idx
        for i in range(n_hyp):
            hyp = {}
            node = self.hyps[T + 1][i]
            hyp['score'] = node.score
            hyp['amscore'] = node.amscore
            word_timestamps = []
            word_hyp = []
            t = T + 1
            is_end = True
            end_timestamp = T
            word = []
            while t > 0:
                if is_end:
                    if node.idx == self.sil_idx or node.idx == self.space_idx:
                        end_timestamp = t - 1
                    else:
                        is_end = False

                if (node.idx == self.sil_idx or node.idx == self.space_idx) \
                    and prev_idx != self.sil_idx and prev_idx != self.space_idx:
                    word_timestamps.append(t - 1)
                    word.reverse()
                    word_hyp.append(word)
                    word = []
                
                if node.idx != self.sil_idx and node.idx != self.space_idx and not node.blank \
                    and (node.idx != prev_idx or prev_blank):
                    word.append(node.idx)

                prev_idx = node.idx
                prev_blank = node.blank
                t -= 1
                node = self.hyps[t][node.parent]
            word_hyp.reverse()
            word_timestamps.reverse()
            hyp['word_timestamps'] = word_timestamps
            hyp['word_hyp'] = word_hyp
            hyp['end_timestamp'] = end_timestamp
            hyps.append(hyp)
        return hyps