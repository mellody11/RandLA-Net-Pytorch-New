# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) on S3DIS and Semantickitti.

**This repository is mainly based on the [repository](https://github.com/qiqihaer/RandLA-Net-pytorch)**.

## Preparation (S3DIS as example)

1. Clone this repository.
2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.
3. Environment:

   ```
   ubuntu 18.04
   python 3.7.16
   torch 1.12.1
   numpy 1.21.5
   torchvision 0.13.1
   scikit-learn 0.22.2
   pandas 1.3.5
   tqdm 4.64.1
   Cython 0.29.33 (Cython is important!)
   ```
4. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).

   ```
   sh compile_op.sh
   ```

   > Attention: please check out *./utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.7-linux-x86_64.egg/* and copy the **.so** file to the parent folder.
   >
   > **Update in 2023.2.23: We provide a **.so** file for python3.7, and you don't need to compile the cpp code if you are using python3.7.**
   >
5. Download the Stanford3dDataset_v1.2_Aligned_Version[ dataset](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1), and preprocess the data:

   > This file will subsample the clouds with 0.04 grid, and we will use these clouds for training.
   >
   > Note: Please change the dataset path in the 'data_prepare_s3dis.py' with your own path.
   >

   ```
   python utils/data_prepare_s3dis.py
   ```
6. As for the data preparation for SemanticKITTI, please refer to [https://github.com/qiqihaer/RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch)

## Train a model (S3DIS as example)

```
  python main_S3DIS.py
```

## Inference on full clouds (S3DIS as example)

```
  python test_S3DIS.py
```

## Results

### S3DIS

We train this network for 100 epoches, and the eval results(after voting) in the Area 5 are as follows: mIoU = 62.59%

```
--------------------------------------------------------------------------------------
62.59 | 91.92 96.32 81.43  0.00 20.59 61.54 55.26 75.03 84.95 56.12 72.33 65.93 52.29 
--------------------------------------------------------------------------------------
```

While [SQN](https://github.com/QingyongHu/SQN) shows the result(mIoU) of RandLA-Net of Area5 is 63.59.

Our results are close to the original paper.

### SemanticKITTI

We train the network for 100 epoches, and the eval results(after voting) in the Seq 08 are as follows: mIoU = 54.62%

```
--------------------------------------------------------------------------------------------------------------------------
54.62 | 93.12 18.31 30.68 79.83 45.59 51.81 70.18  0.00 92.15 41.53 78.42  1.09 87.61 46.32 84.30 58.67 72.12 52.28 33.67 
--------------------------------------------------------------------------------------------------------------------------
```

The checkpoint is in the output folder.

## Model

We provode the pretrained models for S3DIS(Area5) and SemanticKITTI.

S3DIS(Area5): [link](https://drive.google.com/file/d/1VMdJFrPS0TixOKDiRB3AnEQg0JwimAoi/view?usp=drive_link)

SemanticKITTI: [link](https://drive.google.com/file/d/16NYXHN8Yjf_63VCGphP4RhvgTUBLCO3Q/view?usp=drive_link)

## Acknowledgment

This code is mainly based on the [repository](https://github.com/qiqihaer/RandLA-Net-pytorch). Thanks for their efforts.
