# 基于代表片段对比学习的视频异常检测

## Requirements
* Python 3.8
* PyTorch 1.8.0+cu111
* Numpy
* CUDA
* tqdm
* [torchvision](http://pytorch.org/)  
Recommend: the environment can be established by running

```
conda env create -f environment.yaml
```

## Data preparation
Download the [i3d features](link: https://pan.baidu.com/s/1fYAlFoTdcg8BgRdqoLQ2Bg pw: njy2)(https://drive.google.com/file/d/193jToyF8F5rv1SCgRiy_zbW230OrVkuT/view?usp=sharing) and change the "dataset_path" to you/path/data

## Visual Feature Extraction
if you want to extract Visual Feature like this project, you can clone this project([https://github.com/wanboyang/anomaly_feature])


## training & evaluation
we train and evaluate our model in the same process：

```
python main.py --dataset_name shanghaitech --model_name model_sad --Lambda 1_20
```

the above script can complete the training and testing procedure of our model. Specifically, ``--Lambda 1_20`` means $\lambda_1 = 1$ and $\lambda_2 = 20$.
