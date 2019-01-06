#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
import time
import numpy as np
import tensorflow as tf

import crf_model
import data_utils as reader

pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../deep_nlp_python_front/
sys.path.append(pkg_path)

file_path = os.path.dirname(os.path.abspath(__file__))  # ../deep_nlp_python_front/seg_model_python_front/
data_path = os.path.join(file_path, "seg_data")  # path to find corpus vocab file
train_dir = os.path.join(file_path, "seg_model_ckpt")  # path to find model saved checkpoint file

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("seg_data_path", data_path, "data_path")
flags.DEFINE_string("seg_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("seg_scope_name", "seg_var_scope", "Define SEG Variable Scope Name")
flags.DEFINE_integer("max_epoch", 10, "max epochs")

FLAGS = flags.FLAGS

def data_type():
    return tf.float32

def lstm_cell(size):
    return tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0,
                                        state_is_tuple=True,
                                        reuse=tf.get_variable_scope().reuse)

# seg model configuration, set target num, and input vocab_size
class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.05
    learning_rate = 0.005
    max_grad_norm = 5
    hidden_size = 150
    embedding_size = 100
    max_epoch = 5
    max_max_epoch = FLAGS.max_epoch
    stack = False
    keep_prob = 0.8  # There is one dropout layer on input tensor also, don't set lower than 0.9
    lr_decay = 1 / 1.15
    batch_size = 128  # single sample batch
    vocab_size = 49663
    target_num = 4  # SEG tagging tag number for Chinese
    bi_direction = True  # LSTM or BiLSTM


def get_config():
    return LargeConfigChinese()

class Segmenter(object):
    def __init__(self, config, init_embedding = None):
        self.batch_size = batch_size = config.batch_size
        self.embedding_size = config.embedding_size # column
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size # row

        # Define input and target tensors
        self._input_data = tf.placeholder(tf.int32, [batch_size, None], name="input_data")
        self._targets = tf.placeholder(tf.int32, [batch_size, None], name="targets_data")
        self._dicts = tf.placeholder(tf.float32, [batch_size, None], name="dict_data")
        self._seq_len = tf.placeholder(tf.int32, [batch_size], name="seq_len_data")

        with tf.device("/cpu:0"):
            if init_embedding is None:
                self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_size], dtype=data_type())
            else:
                self.embedding = tf.Variable(init_embedding, name="embedding", dtype=data_type())
        inputs = tf.nn.embedding_lookup(self.embedding, self._input_data)
        inputs = tf.nn.dropout(inputs, config.keep_prob)
        inputs = tf.reshape(inputs, [batch_size, -1, 9 * self.embedding_size])
        d = tf.reshape(self._dicts, [batch_size, -1, 16])
        self._loss, self._logits, self._trans = _bilstm_model(inputs, self._targets, d, self._seq_len, config)
        # CRF decode
        self._viterbi_sequence, _ = crf_model.crf_decode(self._logits, self._trans, self._seq_len)
        with tf.variable_scope("train_ops") as scope:
            # Gradients and SGD update operation for training the model.
            self._lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()  # all variables need to train
            # use clip to avoid gradient explosion or gradients vanishing
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), config.max_grad_norm)
            self.optimizer = tf.train.AdamOptimizer(self._lr)
            self._train_op = self.optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

            self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
            self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def dicts(self):
        return self._dicts

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def loss(self):
        return self._loss

    @property
    def logits(self):
        return self._logits

    @property
    def trans(self):
        return self._trans

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def crf_decode_res(self):
        return self._viterbi_sequence

def _bilstm_model(inputs, targets, dicts, seq_len, config):
    '''
    @Use BasicLSTMCell, MultiRNNCell method to build LSTM model
    @return logits, cost and others
    '''
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    target_num = config.target_num  # target output number
    seq_len = tf.cast(seq_len, tf.int32)

    fw_cell = lstm_cell(hidden_size)
    bw_cell = lstm_cell(hidden_size)

    with tf.variable_scope("seg_bilstm"): # like namespace
        # we use only one layer
        (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=tf.float32,
            sequence_length=seq_len,
            scope='layer_1'
        )
        # [batch_size, max_time, cell_fw.output_size]/[batch_size, max_time, cell_bw.output_size]
        output = tf.concat(axis=2, values=[forward_output, backward_output])  # fw/bw dimension is 3
        if config.stack: # False
            (forward_output, backward_output), _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                output,
                dtype=tf.float32,
                sequence_length=seq_len,
                scope='layer_2'
            )
            output = tf.concat(axis=2, values=[forward_output, backward_output])

        output = tf.concat(values=[output, dicts], axis=2)  # add dicts to the end
        # outputs is a length T list of output vectors, which is [batch_size*maxlen, 2 * hidden_size]
        output = tf.reshape(output, [-1, 2 * hidden_size + 16])
        softmax_w = tf.get_variable("softmax_w", [hidden_size * 2 + 16, target_num], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())

        logits = tf.matmul(output, softmax_w) + softmax_b
        logits = tf.reshape(logits, [batch_size, -1, target_num])

    with tf.variable_scope("loss") as scope:
        # CRF log likelihood
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits, targets, seq_len)
        loss = tf.reduce_mean(-log_likelihood)
    return loss, logits, transition_params


def run_epoch(session, model, char_data, tag_data, dict_data, len_data, eval_op, batch_size, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    losses = 0.0
    iters = 0.0

    char_data, tag_data, dict_data, len_data = reader.shuffle(char_data, tag_data, dict_data, len_data)
    xArray, yArray, dArray, lArray = reader.iterator(char_data, tag_data, dict_data, len_data, batch_size)

    for x, y, d, l in zip(xArray, yArray, dArray, lArray):
        fetches = [model.loss, model.logits, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.dicts] = d
        feed_dict[model.seq_len] = l
        loss, logits, _ = session.run(fetches, feed_dict)
        losses += loss
        iters += 1

        if verbose and iters % 50 == 0:
            print("%.3f perplexity: %.3f" %
                  (iters / float(len(xArray)), np.exp(losses / iters / len(xArray))))

    return np.exp(losses / iters)


def evaluate(session, model, char_data, tag_data, dict_data, len_data, eval_op, batch_size, verbose=False):
    """Runs the model on the given data."""
    correct_labels = 0
    total_labels = 0

    xArray, yArray, dArray, lArray = reader.iterator(char_data, tag_data, dict_data, len_data, batch_size)
    yp_wordnum = 0
    yt_wordnum = 0
    cor_num = 0
    for x, y, d, l in zip(xArray, yArray, dArray, lArray):
        fetches = [model.loss, model.logits, model.trans]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.dicts] = d
        feed_dict[model.seq_len] = l
        loss, logits, trans = session.run(fetches, feed_dict)

        for logits_, y_, l_ in zip(logits, y, l):
            logits_ = logits_[:l_]
            y_ = y_[:l_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(logits_, trans)

            yp_wordnum += viterbi_sequence.count(2) + viterbi_sequence.count(3)
            yt_wordnum += (y_ == 2).sum() + (y_ == 3).sum()
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += l_

            start = 0
            for i in range(len(y_)):
                if (y_[i] == 2 or y_[i] == 3):
                    flag = True
                    for j in range(start, i + 1):
                        if y_[j] != viterbi_sequence[j]:
                            flag = False
                    if flag == True:
                        cor_num += 1
                    start = i + 1
    P = cor_num / float(yp_wordnum)
    R = cor_num / float(yt_wordnum)
    F = 2 * P * R / (P + R)
    accuracy = 100.0 * correct_labels / float(total_labels)
    return accuracy, P, R, F


if __name__ == '__main__':
    if not FLAGS.seg_data_path:
        raise ValueError("No data files found in 'data_path' folder")

    print("Begin Loading..")

    raw_data = reader.load_data(FLAGS.seg_data_path)
    train_char, train_tag, train_dict, train_len, dev_char, dev_tag, dev_dict, dev_len, test_char, test_tag, test_dict, test_len, char_vectors, _ = raw_data

    config = get_config()

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.seg_scope_name, reuse=None, initializer=initializer):
            m = Segmenter(config=config, init_embedding=char_vectors)

        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.seg_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.seg_train_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        best_f = 0.0

        # debug model
        # debug_model(session, m, test_char, test_tag, test_dict, test_len, tf.no_op(), config.batch_size)

        for i in range(config.max_max_epoch):
            m.assign_lr(session, config.learning_rate)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_char, train_tag, train_dict, train_len, m.train_op,
                                         config.batch_size, verbose=True)
            dev_acc, dev_precision, dev_recall, dev_f = evaluate(session, m, dev_char, dev_tag, dev_dict, dev_len,
                                                                 tf.no_op(), config.batch_size)
            print("Dev Accuray: %f, Precision: %f, Recall: %f, F score: %f"
                  % (dev_acc, dev_precision, dev_recall, dev_f))
            if dev_f > best_f:
                test_acc, test_precision, test_recall, test_f = evaluate(session, m, test_char, test_tag, test_dict,
                                                                         test_len, tf.no_op(), config.batch_size)
                print("\tTest Accuray: %f, Precision: %f, Recall: %f, F score: %f"
                      % (test_acc, test_precision, test_recall, test_f))
                best_f = dev_f
                checkpoint_path = os.path.join(FLAGS.seg_train_dir, "seg_bilstm.ckpt")
                m.saver.save(session, checkpoint_path)
                print("Model Saved...")

        test_acc, test_precision, test_recall, test_f = evaluate(session, m, test_char, test_tag, test_dict, test_len,
                                                                 tf.no_op(), config.batch_size)
        print("\tTest Accuray: %f, Precision: %f, Recall: %f, F score: %f"% (test_acc, test_precision, test_recall, test_f))

    # 生成测试集的预测结果并将其存储到文件，这里的batch size设置为1
    with tf.Graph().as_default(), tf.Session() as result_session:
        config.batch_size = 1
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope(FLAGS.seg_scope_name, reuse=None, initializer=initializer):
            result_m = Segmenter(config=config, init_embedding=char_vectors)
        ckpt = tf.train.get_checkpoint_state(FLAGS.seg_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            result_m.saver.restore(result_session, tf.train.latest_checkpoint(FLAGS.seg_train_dir))
            print("save model with batch size 1")
            checkpoint_path = os.path.join(FLAGS.seg_train_dir, "seg_bilstm.ckpt")
            result_m.saver.save(result_session, checkpoint_path)
        else:
            print("not found check point files")
