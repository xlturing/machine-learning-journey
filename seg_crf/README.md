# 基于CRF的分词
本子项目利用wapiti来进行序列化任务，原理是利用CRF实现Seq2Seq

## 目录结构
* data为训练、测试、校验文件
* feature为提取的特征文件
* model为保存的CRF模型文件
* pattern存放wapiti提取特征的模板文件
* result为测试集结果输出
* wapiti为wapiti源代码存放地址

## 运行方式
在wapiti目录make编译wapiti可执行文件

python crf_seg_wapiti.py 1(train)/0(test) 1(update)/0(not)


