# 基于Bi-LSTM的分词
利用TensorFlow实现的双向LSTM的分词方式

## 目录结构
seg_data：训练数据以及词典数据
data_utils.py：对于数据的预处理
crf_model.py：tensorflow中抽出的CRF解码部分
seg_model_train.py：模型训练部分
seg_model_test.py：模型预测部分

## 运行方式
安装tensorflow后
python/python3 seg_model_train.py
python/python3 seg_model_test.py
建议使用GPU训练，要不挺慢的
