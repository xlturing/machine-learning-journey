#!/usr/bin/python
# -*- coding:utf-8 -*-

# Reading POS data input_data and target_data

"""Utilities for reading POS train, dev and test files files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode

import collections
import sys, os
import codecs
import re
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

global UNKNOWN
UNKNOWN = "<OOV>"


def _read_file(filename):
    sentences = []  # list(list(str))
    tags = []  # list(list(str))
    chars = []
    infile = codecs.open(filename, encoding='utf-8')

    s = []
    t = []
    for line in infile:
        line = line.strip().split('#')
        if len(line) < 3:
            if len(s) == 0:
                continue
            sentences.append(s)
            tags.append(t)
            s = []
            t = []
        else:
            chars.append(line[1])
            s.append(line[1])
            t.append(line[2].split('_')[0])
    return chars, sentences, tags


def _read_pinyin_dict(filename):
    re_han = re.compile("([\u4E00-\u9FD5]+)", re.U)
    re_skip = re.compile("([^a-zA-Z0-9+#\n])", re.U)

    infile = open(filename, 'r', encoding='utf-8')
    wordset = {}
    wordset["single"] = {}
    wordset["head"] = {}
    wordset["tail"] = {}
    wordset["mid"] = {}

    for line in infile:
        line = line.strip().split('\t')[0]
        blocks = [blk for blk in re_skip.split(line) if len(blk) > 0]

        if len(blocks) == 2:
            word = "".join(blocks)
            if word in wordset["single"]:
                wordset["single"][word] += 1
            else:
                wordset["single"][word] = 1
        elif len(blocks) > 2:
            head = "".join(blocks[:2])
            if head in wordset["head"]:
                wordset["head"][head] += 1
            else:
                wordset["head"][head] = 1

            tail = "".join(blocks[-2:])
            if tail in wordset["tail"]:
                wordset["tail"][tail] += 1
            else:
                wordset["tail"][tail] = 1

            for i in range(1, len(blocks) - 2):
                mid = blocks[i] + blocks[i + 1]
                if mid in wordset["mid"]:
                    wordset["mid"][mid] += 1
                else:
                    wordset["mid"][mid] = 1
    return wordset


def _build_vocab(bigram_filename, vector_filename):
    # bigram from train file
    infile = codecs.open(bigram_filename, encoding='utf-8')
    bigram_chars = set()
    for line in infile:
        com = line.strip().split(' ')
        if len(com[1]) == 2 and int(com[2]) > 10:
            bigram_chars.add(com[1])
    bigram_chars = list(bigram_chars)
    infile.close()

    char_to_id = {}
    char_vectors = []
    if not os.path.isfile(vector_filename):
        char_vectors = None
        # char dictionary
        bigram_chars.append(UNKNOWN)
        counter_char = collections.Counter(bigram_chars)
        count_pairs_char = sorted(counter_char.items(), key=lambda x: (-x[1], x[0]))
        charlist, _ = list(zip(*count_pairs_char))
        char_to_id = dict(zip(charlist, range(len(charlist))))
    else:
        # char unigram dictionary
        infile = codecs.open(vector_filename, encoding='utf-8')
        infile.readline()
        idx = 0
        for line in infile:
            line = line.strip().split(" ")
            char_to_id[line[0]] = idx
            vector = np.asarray(list(map(float, line[1:])), dtype=np.float32)
            char_vectors.append(vector)
            idx += 1
        # char bigram dictionary
        for bigram in bigram_chars:
            mean_vector = np.zeros(100)
            for ch in bigram:
                if ch in char_to_id:
                    mean_vector += char_vectors[char_to_id[ch]]
                else:
                    mean_vector += char_vectors[char_to_id['<OOV>']]
            vector = mean_vector / 2.0
            char_to_id[bigram] = idx
            char_vectors.append(vector)
            idx += 1

    char_vectors = np.asarray(char_vectors, dtype=np.float32)

    # tag dictionary
    taglist = ['B', 'M', 'E', 'S']
    taglist.append(UNKNOWN)
    tag_to_id = dict(zip(taglist, range(len(taglist))))
    return char_to_id, tag_to_id, char_vectors


def _save_vocab(dict, path):
    # save utf-8 code dictionary
    outfile = codecs.open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n"  # unicode
        outfile.write(line)
    outfile.close()


def _read_vocab(path):
    # read utf-8 code
    file = codecs.open(path, encoding='utf-8')
    vocab_dict = {}
    for line in file:
        pair = line.replace("\n", "").split("\t")
        vocab_dict[pair[0]] = int(pair[1])
    return vocab_dict


def load_vocab(data_path):
    char_to_id = _read_vocab(os.path.join(data_path, "char_to_id"))
    tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
    pinyin_dict = _read_pinyin_dict(os.path.join(data_path, "PinyinDict.txt"))
    return char_to_id, tag_to_id, pinyin_dict


def sentence_to_ids(sentence, char_to_id, pinyin_dict):
    sentence.append('<EOS>')
    sentence.append('<EOS>')
    sentence.insert(0, '<BOS>')
    sentence.insert(0, '<BOS>')
    char_idx = []
    dict_value = []
    for i in range(2, len(sentence) - 2):
        for j in range(-2, 3):
            if sentence[i + j] in char_to_id:
                char_idx.append(char_to_id[sentence[i + j]])
            else:
                char_idx.append(char_to_id['<OOV>'])

        for j in range(-2, 2):
            bigram = sentence[i + j] + sentence[i + j + 1]
            if bigram in char_to_id:
                char_idx.append(char_to_id[bigram])
            else:
                char_idx.append(char_to_id['<OOV>'])

            if bigram in pinyin_dict["single"]:
                dict_value.append(1)
                # dict_value.append(pinyin_dict["single"][bigram])
            else:
                dict_value.append(0)
                # dict_value.append("0")

            if bigram in pinyin_dict["head"]:
                dict_value.append(1)
                # dict_value.append(pinyin_dict["head"][bigram])
            else:
                dict_value.append(0)
                # dict_value.append("0")

            if bigram in pinyin_dict["tail"]:
                dict_value.append(1)
                # dict_value.append(pinyin_dict["tail"][bigram])
            else:
                dict_value.append(0)
                # dict_value.append("0")

            if bigram in pinyin_dict["mid"]:
                dict_value.append(1)
                # dict_value.append(pinyin_dict["mid"][bigram])
            else:
                dict_value.append(0)
    return len(sentence) - 4, char_idx, dict_value


def word_ids_to_sentence(data_path, ids):
    tag_to_id = _read_vocab(os.path.join(data_path, "tag_to_id"))
    id_to_tag = {id: tag for tag, id in tag_to_id.items()}
    tagArray = [id_to_tag[i] if i in id_to_tag else id_to_tag[0] for i in ids]
    return tagArray


def _file_to_char_ids(filename, char_to_id, tag_to_id, pinyin_dict):
    _, sentences, tags = _read_file(filename)
    charArray = []
    tagArray = []
    dictArray = []
    lenArray = []
    for sentence, tag in zip(sentences, tags):
        l, char_idx, dict_value = sentence_to_ids(sentence, char_to_id, pinyin_dict)
        lenArray.append(l)
        charArray.append(char_idx)
        dictArray.append(dict_value)
        tagArray.append([tag_to_id[t] if t in tag_to_id else tag_to_id[UNKNOWN] for t in tag])
    return charArray, tagArray, dictArray, lenArray


def load_data(data_path=None):
    """Load POS raw data from data directory "data_path".
    Args: data_path
    Returns:
      tuple (train_data, valid_data, test_data, vocab_size)
      where each of the data objects can be passed to iterator.
    """

    train_path = os.path.join(data_path, "train.txt")
    dev_path = os.path.join(data_path, "dev.txt")
    test_path = os.path.join(data_path, "test.txt")
    vector_path = os.path.join(data_path, "vec100.txt")
    bigram_path = os.path.join(data_path, "words_for_training")
    dict_path = os.path.join(data_path, "PinyinDict.txt")

    char_to_id, tag_to_id, char_vectors = _build_vocab(bigram_path, vector_path)
    pinyin_dict = _read_pinyin_dict(dict_path)
    # Save char_dict and tag_dict
    _save_vocab(char_to_id, os.path.join(data_path, "char_to_id.txt"))
    _save_vocab(tag_to_id, os.path.join(data_path, "tag_to_id.txt"))
    _save_vocab(pinyin_dict["single"], os.path.join(data_path, "pinyin_single.txt"))
    _save_vocab(pinyin_dict["head"], os.path.join(data_path, "pinyin_head.txt"))
    _save_vocab(pinyin_dict["mid"], os.path.join(data_path, "pinyin_mid.txt"))
    _save_vocab(pinyin_dict["tail"], os.path.join(data_path, "pinyin_tail.txt"))
    print("char dictionary size " + str(len(char_to_id)))
    print("tag dictionary size " + str(len(tag_to_id)))

    train_char, train_tag, train_dict, train_len = _file_to_char_ids(train_path, char_to_id, tag_to_id, pinyin_dict)
    print("train dataset: " + str(len(train_char)) + " " + str(len(train_tag)))
    dev_char, dev_tag, dev_dict, dev_len = _file_to_char_ids(dev_path, char_to_id, tag_to_id, pinyin_dict)
    print("dev dataset: " + str(len(dev_char)) + " " + str(len(dev_tag)))
    test_char, test_tag, test_dict, test_len = _file_to_char_ids(test_path, char_to_id, tag_to_id, pinyin_dict)
    print("test dataset: " + str(len(test_char)) + " " + str(len(test_tag)))
    vocab_size = len(char_to_id)
    return (
    train_char, train_tag, train_dict, train_len, dev_char, dev_tag, dev_dict, dev_len, test_char, test_tag, test_dict,
    test_len, char_vectors, vocab_size)


def iterator(char_data, tag_data, dict_data, len_data, batch_size):
    """Iterate on the raw POS tagging file data.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.

    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.

    Raises:
      ValueError: if batch_size or num_steps are too high.
    """

    data_len = len(char_data)
    batch_len = data_len // batch_size
    lArray = []
    xArray = []
    dArray = []
    yArray = []
    for i in range(batch_len):
        if len(len_data[batch_size * i: batch_size * (i + 1)]) == 0:
            continue
        maxlen = max(len_data[batch_size * i: batch_size * (i + 1)])
        l = np.zeros([batch_size], dtype=np.int32)
        x = np.zeros([batch_size, maxlen * 9], dtype=np.int32)
        d = np.zeros([batch_size, maxlen * 16], dtype=np.float32)
        y = np.zeros([batch_size, maxlen], dtype=np.int32)
        l[:len(len_data[batch_size * i:batch_size * (i + 1)])] = len_data[batch_size * i:batch_size * (i + 1)]
        for j, l_j in enumerate(l[:len(len_data[batch_size * i:batch_size * (i + 1)])]):
            x[j][:l_j * 9] = char_data[batch_size * i + j]
            d[j][:l_j * 16] = dict_data[batch_size * i + j]
            y[j][:l_j] = tag_data[batch_size * i + j]
        lArray.append(l)
        xArray.append(x)
        dArray.append(d)
        yArray.append(y)
    return (xArray, yArray, dArray, lArray)


def shuffle(char_data, tag_data, dict_data, len_data):
    char_data = np.asarray(char_data)
    tag_data = np.asarray(tag_data)
    dict_data = np.asarray(dict_data)
    len_data = np.asarray(len_data)
    idx = np.arange(len(len_data))
    np.random.shuffle(idx)

    return (char_data[idx], tag_data[idx], dict_data[idx], len_data[idx])


def main():
    """
    Test load_data method and iterator method
    """
    data_path = "data"
    print("Data Path: " + data_path)
    train_char, train_tag, train_dict, train_len, dev_char, dev_tag, dev_dict, dev_len, test_char, test_tag, test_dict, test_len, char_vectors, _ = load_data(
        data_path)


if __name__ == '__main__':
    main()
