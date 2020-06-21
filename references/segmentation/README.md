# Semantic segmentation

The sample training script was made to train object detection models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip installe -e Holocron/. --upgrade
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation) for you! From there, you can run your training with the following command

```bash
python train.py VOC2012 --model unet3p -b 4 -j 16 --opt radam --lr 1e-5 --sched onecycle --epochs 20
```



## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset using the mean IoU metric.

| Size (px) | Epochs | args                                                         | mean IoU | # Runs |
| --------- | ------ | ------------------------------------------------------------ | -------- | ------ |
| 256       | 20     | VOC2012 --model unet3p -b 4 -j 16 --opt radam --lr 1e-5 --sched onecycle --epochs 20 | 14.17    | 1      |



## Model zoo

| Model  | mean IoU | Param # | MACs | Interpolation | Image size |
| ------ | -------- | ------- | ---- | ------------- | ---------- |
| unet   |          |         |      | bilinear      | 256        |
| unetp  |          |         |      | bilinear      | 256        |
| unetpp |          |         |      | bilinear      | 256        |
| unet3p |          |         |      | bilinear      | 256        |

