# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import codecs
import data_utils as reader
import numpy as np
import time


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(file_path, "seg_data/")
    check_point_dir_path = "seg_model_ckpt/"
    # op names
    input_data_name = "seg_var_scope/input_data:0"
    targets_data_name = "seg_var_scope/targets_data:0"
    dict_data_name = "seg_var_scope/dict_data:0"
    seq_len_name = "seg_var_scope/seq_len_data:0"

    crf_decode_name = "seg_var_scope/ReverseSequence_1:0"

    # load data sets
    start_time = time.clock()
    raw_data = reader.load_data(data_path)
    train_char, train_tag, train_dict, train_len, dev_char, dev_tag, dev_dict, dev_len, test_char, test_tag, test_dict, test_len, char_vectors, _ = raw_data

    # load dictionary
    tag_id_file = codecs.open("seg_data/tag_to_id.txt", mode='r', encoding='utf-8')

    tag_to_id = dict()
    for line in tag_id_file:
        line_list = line.strip().split('\t')
        tag_to_id[line_list[0]] = int(line_list[1])
    id_to_tag = dict()
    for k,v in tag_to_id.items():
        id_to_tag[v] = k


    # prepare test data
    xArray, yArray, dArray, lArray = reader.iterator(test_char, test_tag, test_dict, test_len, 1)#batch size 1
    end_time = time.clock()
    print("data load time:", end_time-start_time)

    # open files
    test_reader = codecs.open("seg_data/test_for_python.txt", mode='r', encoding='utf-8')
    test_sentences = []
    s = []
    for line in test_reader:
        line = line.strip()
        if len(line) == 0:
            if len(s) == 0:
                continue
            test_sentences.append(s)
            s = []

        else:
            s.append(line)
    print("len sentences:", len(test_sentences))
    test_writer = codecs.open("seg_data/test_data_python_res.txt", mode='w', encoding='utf-8')
    # load model and predict
    with tf.Session() as sess:
        # load model
        start_time = time.clock()
        saver = tf.train.import_meta_graph(check_point_dir_path+'seg_bilstm.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(check_point_dir_path))
        graph = tf.get_default_graph()
        # feed ops
        input_data = graph.get_tensor_by_name(input_data_name)
        dict_data = graph.get_tensor_by_name(dict_data_name)
        seq_len = graph.get_tensor_by_name(seq_len_name)
        target_data = graph.get_tensor_by_name(targets_data_name)

        # fetch ops
        crf_decode = graph.get_tensor_by_name(crf_decode_name)
        end_time = time.clock()
        print("model load time:", end_time-start_time)
        start_time = time.clock()
        # session run
        for x, y, d, l, sents in zip(xArray, yArray, dArray, lArray, test_sentences):
            fetches = [crf_decode]
            feed_dict = dict()
            feed_dict[input_data] = x
            feed_dict[dict_data] = d
            feed_dict[target_data] = y
            feed_dict[seq_len] = l
            crf_decode_res = sess.run(fetches, feed_dict)

            # save result
            for char, res_tag in zip(sents, crf_decode_res[0][0]):
                test_writer.write(str(char)+"\t"+str(id_to_tag[int(res_tag)])+"\n")
            test_writer.write("\n")
        end_time = time.clock()
        print("model run time:", end_time-start_time)