# 机器学习课程大作业 -- DL入门

本项目是一个语音情感识别项目，使用多种的预处理方法，使用多种模型，实现了语音情感识别。

## 作业任务
### 完成项目环境初始化
1. 完成项目环境初始化，包括安装Pytorch、mser库等。
2. 准备数据，完成数据列表的生成。
3. 提取特征，完成特征的提取。
### 在项目中增加DNN/CNN/LSTM/Transformer四种类型的模型
1. 使用base model完成初步训练
2. 在models文件夹中增加DNN/CNN/LSTM/Transformer四种类型的模型
3. 完成四种模型的训练
4. 完成四种模型的评估
5. 录制自己的8类情感语音数据["中性", "平静", "快乐", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶"]，每种5句，并将其放于“/share/MER_Data/学号”下
6. 完成四种模型的在自己声音下的评估预测
### 完成研究报告
1. 完成研究报告，包括项目背景、项目目的、项目方法、项目结果、项目结论等。
2. 探讨相关工作的问题
3. 提出自己的解决方案


## 安装环境

 - 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。
```shell
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

 - 安装mser库。
 
使用pip安装，命令如下：
```shell
python -m pip install mser -U -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 准备数据
下载：https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip ，并将其放于 dataset文件夹下。也可以直接从/share/DL_data/Audio_Speech_Actors_01-24.zip 从拷贝至 dataset文件夹。

然后执行`create_data.py`里面的`create_ravdess_list('dataset/Audio_Speech_Actors_01-24', 'dataset')`函数即可生成数据列表，同时也生成归一化文件，具体看代码。

```shell
python create_data.py
```

如果自定义数据集，可以按照下面格式，`audio_path`为音频文件路径，用户需要提前把音频数据集存放在`dataset/audio`目录下，每个文件夹存放一个类别的音频数据，每条音频数据长度在3秒左右，如 `dataset/audio/angry/······`。`audio`是数据列表存放的位置，生成的数据类别的格式为 `音频路径\t音频对应的类别标签`，音频路径和标签用制表符 `\t`分开。读者也可以根据自己存放数据的方式修改以下函数。

执行`create_data.py`里面的`get_data_list('dataset/audios', 'dataset')`函数即可生成数据列表，同时也生成归一化文件，具体看代码。
```shell
python create_data.py
```

生成的列表是长这样的，前面是音频的路径，后面是该音频对应的标签，从0开始，路径和标签之间用`\t`隔开。
```shell
dataset/Audio_Speech_Actors_01-24/Actor_13/03-01-01-01-02-01-13.wav	0
dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-02-01-01-01-01.wav	1
dataset/Audio_Speech_Actors_01-24/Actor_01/03-01-03-02-01-01-01.wav	2
```

**注意：** `create_data.py`里面的`create_standard('configs/base.yml')`函数必须要执行的，这个是生成归一化的文件。


# 提取特征

在训练过程中，首先是要读取音频数据，然后提取特征，最后再进行训练。其中读取音频数据、提取特征也是比较消耗时间的，所以我们可以选择提前提取好取特征，训练模型的是就可以直接加载提取好的特征，这样训练速度会更快。这个提取特征是可选择，如果没有提取好的特征，训练模型的时候就会从读取音频数据，然后提取特征开始。提取特征步骤如下：

1. 执行`extract_features.py`，提取特征，特征会保存在`dataset/features`目录下，并生成新的数据列表`train_list_features.txt`和`test_list_features.txt`。

```shell
python extract_features.py --configs=configs/base.yml --save_dir=dataset/features
```

2. 修改配置文件，将`dataset_conf.train_list`和`dataset_conf.test_list`修改为`train_list_features.txt`和`test_list_features.txt`。


## 训练

训练有两个方法，第一个是提前提取特征，保持在本地，然后在进行训练，这种方法的好处就是训练特别快，因为本项目的特征提取方法比较慢，如果在训练中要提取特征，那么训练会很慢，缺点是没办法使用随机数据增强。第二种就是在训练过程中提取特征，这种好处是可以使用随机数据增强，缺点是训练比较慢。

 - 提取特征（可选），执行`extract_features.py`程序即可，特征提取完成需要修改`configs/base.yml`里面的`train_list`和`test_list`，将它们修改为新生成的数据列表路径。

```shell
python extract_features.py --configs=configs/base.yml
```

输出日志：
```
·······
100%████████████████████████████| 1290/1290 [01:39<00:00, 12.99it/s]
[2024-02-03 14:57:00.699338 INFO   ] trainer:get_standard_file:136 - 归一化文件保存在：dataset/standard.m
[2024-02-03 14:57:00.700046 INFO   ] featurizer:__init__:23 - 使用的特征方法为 Emotion2Vec
100%|████████████████████████████| 1290/1290 [01:36<00:00, 13.40it/s]
[2024-02-03 14:58:36.941253 INFO   ] trainer:extract_features:162 - dataset/train_list.txt列表中的数据已提取特征完成，新列表为：dataset/train_list_features.txt
100%|██████████████████████████████| 150/150 [00:11<00:00, 13.52it/s]
[2024-02-03 14:58:48.036661 INFO   ] trainer:extract_features:162 - dataset/test_list.txt列表中的数据已提取特征完成，新列表为：dataset/test_list_features.txt
```

不管是否提前提取特征，接着都可以开始训练模型了，创建 `train.py`。配置文件里面的参数一般不需要修改，但是这几个是需要根据自己实际的数据集进行调整的，首先最重要的就是分类大小`dataset_conf.num_class`，这个每个数据集的分类大小可能不一样，根据自己的实际情况设定。然后是`dataset_conf.batch_size`，如果是显存不够的话，可以减小这个参数。

```shell
# 单卡训练
CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/base.yml
```

# 评估

执行下面命令执行评估。

```shell
python eval.py --configs=configs/base.yml --resume_model=models/BaseModel_Emotion2Vec/best_model
```

评估输出如下：
```shell
[2024-02-03 15:13:25.469242 INFO   ] trainer:evaluate:461 - 成功加载模型：models/BiLSTM_Emotion2Vec/best_model/model.pth
100%|██████████████████████████████| 150/150 [00:00<00:00, 1281.96it/s]
评估消耗时间：1s，loss：0.61840，accuracy：0.87333
```

评估会出来输出准确率，还保存了混淆矩阵图片，保存路径`output/images/`。


# 预测

在训练结束之后，我们得到了一个模型参数文件，我们使用这个模型预测音频。

```shell
python infer.py --audio_path=inference/test.wav --model_path=models/BaseModel_Emotion2Vec/best_model
```

输出如下：
```
成功加载模型参数：models/BiLSTM_Emotion2Vec/best_model/model.pth
[2024-07-02 19:48:42.864262 INFO   ] emotion2vec_predict:__init__:27 - 成功加载模型：models/iic/emotion2vec_base
音频：inference/test.wav 的预测结果标签为：悲伤，得分：0.88658
```
