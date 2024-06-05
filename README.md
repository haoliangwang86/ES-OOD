# ES-OOD Overview

![image](https://github.com/haoliangwang86/ES-OOD/assets/71032219/d87fe5f4-18eb-4a55-a78e-01c0810a8f30)
An overview of our proposed Early Stopping OOD detection (\sysname{}) framework that utilizes all intermediate representations and an early stopping strategy for efficient and effective OOD detection in DNNs.

# Requirements
* Pyhton 3.7.5
* Pytorch 1.9.0
* CUDA 10.1

Please use the following codes to install the full requirements:
```python
pip install -r requirements.txt
```

# Pre-trained models
|   Model  | CIFAR10 | CIFAR100 |
|:--------:|:-------:|:--------:|
|    VGG   |  93.94%  |   74.13%  |
|  ResNet  |  94.67%  |   75.02%  |
| DenseNet |  95.06%  |   77.18%  |

# Running the codes
## 1. Train the backbone models
If you wish to train a new backbone model, run the following:
```python
python train_backbone_model.py --model vgg16 --dataset cifar10
```

for the backbone models, choose from:
* vgg16
* resnet34
* densenet100

for the training dataset, choose from:
* cifar10
* cifar100

## 2. Download the OOD datasets
Download the following datasets:
* LSUN test set: https://github.com/fyu/lsun
* Tiny ImageNet: https://image-net.org/index.php
* DTD dataset: https://www.robots.ox.ac.uk/~vgg/data/dtd/

save the unzipped files in ./data folder


## 3. Generate all the datasets
Generate the InD and OOD datasets:
```python
python generate_datasets.py
```

## 4. Save the intermedia outputs
```python
python save_inter_outputs.py --model vgg16 --ind cifar10
```

## 5. Train OOD detectors
```python
python train_ood_detectors.py --model vgg16 --ind cifar10
```

## 6. Test: OOD detection (with early stopping)
```python
python detect_oods_early_stopping.py --model vgg16 --ind cifar10 --thr 0.99 --k 2 --folder results-thr-0.99-k-2
```

## 7. Test: OOD detection (without early stopping)
```python
python detect_oods.py --model vgg16 --ind cifar10
```

## 8. Co-train models (optional)
```python
python co_train.py --model vgg16 --dataset cifar10
```
