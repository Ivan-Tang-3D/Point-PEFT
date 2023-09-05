# Point-PEFT: Parameter-Efficient Fine-Tuning for 3D Pre-trained Models

Official implementation of ['Point-PEFT: Parameter-Efficient Fine-Tuning for 3D Pre-trained Models']().

## News
* 📣 We release the code of Point-M2AE with Point-PEFT.

## Introduction

Comparison with existing 3D pre-trained models on the PB-T50-RS spilt of ScanObjectNN:
| Method | Parameters | PB-T50-RS|
| :-----: | :-----: |:-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M | 83.1 %| 
| **With Point-PEFT** | **0.6M** | **85.0%**|
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | 22.1M | 85.2%|
| **With Point-PEFT** | **0.7M** | **85.5%**|
| **Point-MAE-aug** | **22.1M** | **88.1%**|
| **With Point-PEFT** | **0.7M** | **89.1%**|
| [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE)| 12.9M | 86.4%|
| **With Point-PEFT** | **0.7M** | **86.4%**|
| **Point-M2AE-aug** | **12.9M** | **88.1%**|
| **With Point-PEFT** | **0.7M** | **88.2%**|

Comparison with existing 3D pre-trained models on the ModelNet40 without voting method:
| Method | Parameters | Acc|
| :-----: | :-----: |:-----:|
| [Point-BERT](https://github.com/lulutang0608/Point-BERT) | 22.1M | 92.7 %| 
| **With Point-PEFT** | **0.6M** | **93.4%**|
| [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) | 22.1M | 93.2%|
| **With Point-PEFT** | **0.8M** | **94.2%**|
| [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE)| 15.3M | 93.4%|
| **With Point-PEFT** | **0.6M** | **94.1%**|

We propose an alternative to obtain superior 3D representations from 2D pre-trained models via **I**mage-to-**P**oint Masked Autoencoders, named as **I2P-MAE**. By self-supervised pre-training, we leverage the well learned 2D knowledge to guide 3D masked autoencoding, which reconstructs the masked point tokens with an encoder-decoder architecture. Specifically, we conduct two types of image-to-point learning schemes: 2D-guided masking and 2D-semantic reconstruction. In this way, the 3D network can effectively inherit high-level 2D semantics learned from rich image data for discriminative 3D modeling.

<div align="center">
  <img src="pipeline.png"/>
</div>

## I2P-MAE Models

### Pre-training
Guided by pre-trained CLIP on ShapeNet, I2P-MAE is evaluated by **Linear SVM** on ModelNet40 and ScanObjectNN (OBJ-BG split) datasets, without downstream fine-tuning:
| Task | Dataset | Config | MN40 Acc.| OBJ-BG Acc.| Ckpts | Logs |   
| :-----: | :-----: |:-----:| :-----: | :-----:| :-----:|:-----:|
| Pre-training | ShapeNet |[i2p-mae.yaml](./cfgs/pre-training/i2p-mae.yaml)| 93.35% | 87.09% | [pre-train.pth](https://drive.google.com/file/d/1TYKHdLwu9DKLgsnvsY4fpgpNowCHErFZ/view?usp=share_link) | [log](https://drive.google.com/file/d/11kkgTQoUJVLYKk1Xbo0XtQPCqh50my-G/view?usp=share_link) |

### Fine-tuning
Synthetic shape classification on ModelNet40 with 1k points:
| Task  | Config | Acc.| Vote| Ckpts | Logs |   
| :-----: | :-----:| :-----:| :-----: | :-----:|:-----:|
| Classification | [modelnet40.yaml]()|93.67%| 94.06% | [modelnet40.pth]() | [modelnet40.log]() |

Real-world shape classification on ScanObjectNN:
| Task | Split | Config | Acc.| Ckpts | Logs |   
| :-----: | :-----:|:-----:| :-----:| :-----:|:-----:|
| Classification | PB-T50-RS|[scan_pb.yaml]() | 90.11%| [scan_pd.pth]() | [scan_pd.log]() |
| Classification |OBJ-BG| [scan_obj-bg.yaml]() | 94.15%| - | - |
| Classification | OBJ-ONLY| [scan_obj.yaml]() | 91.57%| - | - |


## Requirements

### Installation
Create a conda environment and install basic dependencies:
```bash
git clone https://github.com/ZrrSkywalker/I2P-MAE.git
cd I2P-MAE

conda create -n i2pmae python=3.7
conda activate i2pmae

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
# e.g., conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

pip install -r requirements.txt
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
### Datasets
For pre-training and fine-tuning, please follow [DATASET.md](https://github.com/lulutang0608/Point-BERT/blob/master/DATASET.md) to install ShapeNet, ModelNet40, ScanObjectNN, and ShapeNetPart datasets, referring to Point-BERT. Specially for Linear SVM evaluation, download the official [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset and put the unzip folder under `data/`.

The final directory structure should be:
```
│I2P-MAE/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──modelnet40_ply_hdf5_2048/  # Specially for Linear SVM
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──shapenetcore_partanno_segmentation_benchmark_v0_normal/
├──...
```

## Get Started

### Pre-training
I2P-MAE is pre-trained on ShapeNet dataset with the config file `cfgs/pre-training/i2p-mae.yaml`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pre-training/i2p-mae.yaml --exp_name pre-train
```

To evaluate the pre-trained I2P-MAE by **Linear SVM**, create a folder `ckpts/` and download the [pre-train.pth]() into it. Use the configs in `cfgs/linear-svm/` and indicate the evaluation dataset by `--test_svm`.

For ModelNet40, run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/linear-svm/modelnet40.yaml --test_svm modelnet40 --exp_name test_svm --ckpts ./ckpts/pre-train.pth
```
For ScanObjectNN (OBJ-BG split), run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/linear-svm/scan_obj-bg.yaml --test_svm scan --exp_name test_svm --ckpts ./ckpts/pre-train.pth
```

### Fine-tuning
Please create a folder `ckpts/` and download the [pre-train.pth]() into it. The fine-tuning configs are in `cfgs/fine-tuning/`.

For ModelNet40, run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/modelnet40.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```

For the three splits of ScanObjectNN, run:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_pb.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_obj.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/fine-tuning/scan_obj-bg.yaml --finetune_model --exp_name finetune --ckpts ckpts/pre-train.pth
```


## Acknowledgement
This repo benefits from [Point-M2AE](https://github.com/ZrrSkywalker/Point-M2AE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), and [CLIP](https://github.com/openai/CLIP). Thanks for their wonderful works.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.

## Contact
If you have any question about this project, please feel free to contact zhangrenrui@pjlab.org.cn.

