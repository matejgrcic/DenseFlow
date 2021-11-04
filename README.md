# Densely connected normalizing flows

This repository is the official implementation of **NeurIPS 2021** paper [Densely connected normalizing flows](https://arxiv.org/abs/2106.04627).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/densely-connected-normalizing-flows/image-generation-on-imagenet-32x32)](https://paperswithcode.com/sota/image-generation-on-imagenet-32x32?p=densely-connected-normalizing-flows)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/densely-connected-normalizing-flows/image-generation-on-imagenet-64x64)](https://paperswithcode.com/sota/image-generation-on-imagenet-64x64?p=densely-connected-normalizing-flows)
##  Setup

- CUDA 11.1 
- Python 3.8

```
pip install -r requirements.txt
pip install -e .
```
## Training
  
```
cd ./experiments/image
```
CIFAR-10:
```
python train.py --epochs 400 --batch_size 64 --optimizer adamax --lr 1e-3  --gamma 0.9975 --warmup 5000  --eval_every 1 --check_every 10 --dataset cifar10 --augmentation eta --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_74_10
```
```
python train_more.py --model ./log/cifar10_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 420
```
ImageNet32:
```
python train.py --epochs 20 --batch_size 64 --optimizer adamax --lr 1e-3  --gamma 0.95 --warmup 5000  --eval_every 1 --check_every 10 --dataset imagenet32 --augmentation eta --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_74_10
```
```
python train_more.py --model ./log/imagenet32_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 22
```
ImageNet64:
```
python train.py --epochs 10 --batch_size 32 --optimizer adamax --lr 1e-3  --gamma 0.95 --warmup 5000  --eval_every 1 --check_every 10 --dataset imagenet64 --augmentation eta --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_74_10
```
```
python train_more.py --model ./log/imagenet64_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 11
```
CelebA:
```
python train.py --epochs 50 --batch_size 32 --optimizer adamax --lr 1e-3  --gamma 0.95 --warmup 5000  --eval_every 1 --check_every 10 --dataset celeba --augmentation horizontal_flip --block_conf 6 4 1 --layers_conf  5 6 20  --layer_mid_chnls 48 48 48 --growth_rate 10  --name DF_74_10
```
```
python train_more.py --model ./log/celeba_8bit/densenet-flow/expdecay/DF_74_10 --new_lr 2e-5 --new_epochs 55
```
**Note:** Download instructions for ImageNet and CelebA are defined in `denseflow/data/datasets/image/{dataset}.py`
## Evaluation

CIFAR-10:
```
python eval_loglik.py --model PATH_TO_MODEL --k 1000 --kbs 50
```
ImageNet32:
```
python eval_loglik.py --model PATH_TO_MODEL --k 200 --kbs 50
```
ImageNet64 and CelebA:
```
python eval_loglik.py --model PATH_TO_MODEL --k 200 --kbs 25
```

## Model weights
Model weights are stored [here](https://drive.google.com/file/d/1CAX-TV4ZTtNbb57UYTn6j-rY7CQFpocp/view?usp=sharing).

## Samples generation
Generated samples are stored in `PATH_TO_MODEL/samples`
```
python eval_sample.py --model PATH_TO_MODEL
```
**Note:** `PATH_TO_MODEL` has to contain `check` directory.

### ImageNet 32x32

![Alt text](assets/ImageNet32.png?raw=true)

### ImageNet 64x64

![Alt text](assets/ImageNet64.png?raw=true)

### CelebA

![Alt text](assets/CelebA.png?raw=true)


### Acknowledgements
Significant part of this code benefited from SurVAE [1] [code implementation](https://github.com/didriknielsen/survae_flows), available under MIT license.


### References
[1] Didrik Nielsen, Priyank Jaini, Emiel Hoogeboom, Ole Winther, and Max Welling. Survae flows: Surjections to bridge the gap between vaes and flows. InAdvances in Neural Information Processing Systems 33. Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020.
