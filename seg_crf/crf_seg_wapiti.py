#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import time


# extract segment freature from sentence list
# C-2/C-1/C1/C2/C0C1/C-1C0/C-2C-1/C-1C0C1
def extract_seg_feature(sentencelist):
    sentence_feature_list = []
    for i in range(len(sentencelist)):
        tokenlist = sentencelist[i]
        token_feature_list = []
        for j in range(len(tokenlist)):
            featurelist = [("cur", tokenlist[j][1])]
            if j >= 2:
                featurelist.append(("pre2", tokenlist[j - 2][1]))
            else:
                featurelist.append(("pre2", "<UNK>"))
            if j >= 1:
                featurelist.append(("pre1", tokenlist[j - 1][1]))
            else:
                featurelist.append(("pre1", "<UNK>"))
            if j < len(tokenlist) - 1:
                featurelist.append(("next1", tokenlist[j + 1][1]))
            else:
                featurelist.append(("next1", "<UNK>"))
            if j < len(tokenlist) - 2:
                featurelist.append(("next2", tokenlist[j + 2][1]))
            else:
                featurelist.append(("next2", "<UNK>"))
            featurelist.append(("label", tokenlist[j][2].split("_")[0]))
            token_feature_list.append(featurelist)
        sentence_feature_list.append(token_feature_list)
    return sentence_feature_list


def dump_crf_feature(sentence_feature_list, outfd):
    for i in range(len(sentence_feature_list)):
        token_feature_list = sentence_feature_list[i]
        for j in range(len(token_feature_list)):
            featurelist = token_feature_list[j]
            outfd.write(
                "\t".join([str(feature[1]) if len(str(feature[1])) else "<UNK>" for feature in featurelist]) + "\n")
        outfd.write("\n")


def preprocess(path):
    sentencelist = []
    with open(path) as f:
        tokenlist = []
        line = f.readline()
        while line:
            token = line.strip().split('#')
            tokenlist.append(token)
            line = f.readline()
            if line.strip() == '':
                sentencelist.append(tokenlist)
                tokenlist = []
                line = f.readline()
    return sentencelist


def train(new_data_model_path=None):
    cmd = "%s/wapiti train %s -t 5 -e 0.02 -p %s -d %s %s %s" % \
          ("wapiti",
           "-1 2 -a sgd-l1",
           "pattern/pat.txt",
           "feature/seg_fea.dev",
           "feature/seg_fea.train",
           "model/seg.model")
    print(cmd)
    os.system(cmd)


def test():
    cmd = "%s/wapiti label -s -c -p -m %s %s %s.out" % \
          ("wapiti",
           "model/seg.model",
           "feature/seg_fea.test",
           "result/seg_rs")
    print(cmd)
    os.system(cmd)


if __name__ == "__main__":
    root_path = ""
    trainlist = preprocess(root_path + 'data/train.txt')
    devlist = preprocess(root_path + 'data/dev.txt')
    testlist = preprocess(root_path + 'data/test.txt')
    if len(sys.argv) < 2:
        print("usage python *.py 1(train)/0(test) 1(update)/0(not)")
        exit(-1)
    if len(sys.argv) > 2 and sys.argv[2] == '1':
        dump_crf_feature(extract_seg_feature(trainlist), open(root_path + "feature/seg_fea.train", "w"))
        dump_crf_feature(extract_seg_feature(devlist), open(root_path + "feature/seg_fea.dev", "w"))
        dump_crf_feature(extract_seg_feature(testlist), open(root_path + "feature/seg_fea.test", "w"))
    if sys.argv[1] == '1':
        curtime = time.time()
        train()
        print(time.time() - curtime)
    else:
        test()
