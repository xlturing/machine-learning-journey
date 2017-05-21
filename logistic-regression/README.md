# Logistic Regression的简单实现
本程序利用传统的batch gradient descent进行训练，并且利用了进行了[matplotlib.cpp](https://github.com/lava/matplotlib-cpp)画图处理，该lib将Python当中的画图包进行了封装，使得其能够在c++中调用

## 运行
需要添加c++11的特性，并且需要添加-lboost_regex
按照[matplotlib.cpp](https://github.com/lava/matplotlib-cpp)中的方式安装好python后
直接在Debug目录进行make，产生可执行文件运行便可以看到生成的图片。