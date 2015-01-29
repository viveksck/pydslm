# -*- coding: utf-8 -*-
import gensim
from copy import deepcopy
from gensim.models.word2vec import Word2Vec, train_sentence_sg, Vocab
from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from six import iteritems, itervalues, string_types
from six.moves import xrange

import logging
logger = logging.getLogger("gensim.models.word2vec")

def train_sentence_sg_ds(model, sentence_signal, alpha, work = None):
        """
        Update skip-gram model by training on a single sentence.
        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary followed by a signal. Called internally from `Word2Vec.train()`.
        This is the non-optimized, Python version. If you have cython installed, gensim
        will use the optimized version from word2vec_inner instead.
        """

        labels = []
        if model.negative:
            # precompute negative labels
            labels = zeros(model.negative + 1)
            labels[0] = 1.0

        sentence, signal = model.extract_sentence_and_signal(sentence_signal)
        active_features = model.get_active_features(signal)

        for pos, word in enumerate(sentence):
            if word is None:
                continue  # OOV word in the input sentence => skip
            reduced_window = random.randint(model.window)  # `b` in the original word2vec code

            # now go over all words from the (reduced) window, predicting each
            # one in turn
            start = max(0, pos - model.window + reduced_window)
            for pos2, word2 in enumerate(sentence[start : pos + model.window + 1 - reduced_window], start):
                # don't train on OOV words and on the `word` itself
                if word2 and not (pos2 == pos):
                    train_sg_ds_pair(model, word, word2, alpha, labels, train_w1=True, train_w2=True, active_features = active_features)

        return len([word for word in sentence if word is not None])

def train_sg_ds_pair(model, word, word2, alpha, labels, train_w1=True, train_w2=True, active_features=["MAIN"]):

    # Obtain the embedding for the word2 by summing up MAIN embedding and
    # active features delta embeddings. active_features list always contains
    # MAIN
    l1 = zeros(model.embeddings_map["MAIN"][word2.index].shape)
    for feature in active_features:
        l1 = l1 + model.embeddings_map[feature][word2.index]

    neu1e = zeros(l1.shape)
    if model.hs:
        # work on the entire tree at once, to push as much work into numpy's C
        # routines as possible (performance)
        l2a = deepcopy(model.syn1[word.point])  # 2d matrix, codelen x layer1_size
        fa = 1.0 / (1.0 + exp(-dot(l1, l2a.T)))  # propagate hidden -> output
        ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        if train_w1:
            model.syn1[word.point] += outer(ga, l1)  # learn hidden -> output
        neu1e += dot(ga, l2a)  # save error

    if model.negative:
        # use this word (label = 1) + `negative` other random words not from
        # this sentence (label = 0)
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.table[random.randint(model.table.shape[0])]
            if w != word.index:
                word_indices.append(w)
        l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
        fb = 1. / (1. + exp(-dot(l1, l2b.T)))  # propagate hidden -> output
        gb = (labels - fb) * alpha  # vector of error gradients multiplied by the learning rate
        if train_w1:
            model.syn1neg[word_indices] += outer(gb, l1)  # learn hidden -> output
        neu1e += dot(gb, l2b)  # save error

    if train_w2:
        # Back prop the error to all constituent embeddings. MAIN and active.
        for feature in active_features:
            model.embeddings_map[feature][word2.index] += neu1e  # learn input -> hidden

    return neu1e

class DTWord2Vec(Word2Vec):

    def extract_sentence_and_signal(self, sentence_signal):
        sentence, signal = sentence_signal[:-1], sentence_signal[-1]
        return sentence, signal

    def get_active_features(self, signal_point):
        """ Return the set of active features. """
        return ["MAIN", signal_point]

    def _vocab_from(self, sentences):
        """ Construct the vocabulary. """
        self.signals = set([])
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence_signal in enumerate(sentences):
            sentence, signal = self.extract_sentence_and_signal(sentence_signal)
            self.signals.add(signal)
            if sentence_no % 10000 == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" % (sentence_no, total_words, len(vocab)))
            for word in sentence:
                total_words += 1
                if word in vocab:
                    vocab[word].count += 1
                else:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i word types from a corpus of %i words and %i sentences" % (len(vocab), total_words, sentence_no + 1))
        return vocab

    def _get_job_words(self, alpha, work, job, neu1):
        return sum(train_sentence_sg_ds(self, sentence, alpha, work) for sentence in job)

    def _prepare_sentences(self, sentences):
        for sentence_signal in sentences:
            sentence, signal = self.extract_sentence_and_signal(sentence_signal)
            # avoid calling random_sample() where prob >= 1, to speed things up
            # a little:
            sampled = [self.vocab[word] for word in sentence if word in self.vocab and (self.vocab[word].sample_probability >= 1.0 or  self.vocab[word].sample_probability >= random.random_sample())]

            sampled_signal = sampled + [signal]
            yield sampled_signal 

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        logger.info("resetting layer weights") 
        self.embeddings_map= {}
        self.embeddings_map["MAIN"] = empty((len(self.vocab), self.layer1_size), dtype=REAL)

        for signal in self.signals:
            self.embeddings_map[signal] = zeros((len(self.vocab), self.layer1_size), dtype=REAL)

        # randomize weights vector by vector, rather than materializing a huge
        # random matrix in RAM at once
        for i in xrange(len(self.vocab)):
            # construct deterministic seed from word AND seed argument
            # Note: Python's built in hash function can vary across versions of
            # Python
            random.seed(uint32(self.hashfxn(self.index2word[i] + str(self.seed))))
            self.embeddings_map["MAIN"][i] = (random.rand(self.layer1_size) - 0.5) / self.layer1_size

        if self.hs:
            self.syn1 = zeros((len(self.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.vocab), self.layer1_size),dtype=REAL)
        self.syn0norm = None
        return

    def save_word2vec_format(self, fname, fvocab=None, binary=False):
        """
        Store the input-hidden weight matrix in the same format used by the original
        C word2vec-tool, for compatibility.
        """
        if fvocab is not None:
            logger.info("Storing vocabulary in %s" % (fvocab))
            with utils.smart_open(fvocab, 'wb') as vout:
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item: -item[1].count):
                    vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))

        logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.layer1_size, fname))
        assert (len(self.vocab), self.layer1_size) == self.embeddings_map["MAIN"].shape
        with utils.smart_open(fname, 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % self.embeddings_map["MAIN"].shape))

            for signal in ['MAIN'] + sorted(list(self.signals)):
                emb_vecs = self.embeddings_map[signal]
                # store in sorted order: most frequent words at the top
                for word, vocab in sorted(iteritems(self.vocab), key=lambda item:-item[1].count):
                    row = emb_vecs[vocab.index]
                    if binary:
                        fout.write(utils.to_utf8(signal) + b" " + utils.to_utf8(word) + b" " + row.tostring())
                    else:
                        fout.write(utils.to_utf8("%s %s %s\n" % (signal, word, ' '.join("%f" % val for val in row))))

