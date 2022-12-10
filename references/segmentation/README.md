# Semantic segmentation

The sample training script was made to train object detection models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip install -e "Holocron/.[training]"
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation) for you! From there, you can run your training with the following command

```bash
python train.py VOC2012 --arch unet3p -b 4 -j 16 --opt radam --lr 1e-5 --sched onecycle --epochs 20
```



## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset using the mean IoU metric.

| Size (px) | Epochs | args                                                         | mean IoU | # Runs |
| --------- | ------ | ------------------------------------------------------------ | -------- | ------ |
| 256       | 200    | VOC2012 --arch unet_rexnet13 -b 16 --loss label_smoothing --opt adamp --device 0 --lr 2e-3 --epochs 200 | 32.14    | 1      |
| 256       | 20     | VOC2012 --arch unet3p -b 4 -j 16 --opt radam --lr 1e-5 --sched onecycle --epochs 20 | 14.17    | 1      |



## Model zoo

| Model         | mean IoU | Param # | MACs | Interpolation | Image size |
| ------------- | -------- | ------- | ---- | ------------- | ---------- |
| unet          |          | 18.11M |      | bilinear      | 256        |
| unetp         |          | 28.28M  |      | bilinear      | 256        |
| unetpp        |          | 29.54M  |      | bilinear      | 256        |
| unet3p        |          | 26.93M  |      | bilinear      | 256    |
| unet_tvvgg11  |          | 32.17M |      | bilinear      | 256        |
| unet_tvresnet34 |     | 36.25M |      | bilinear      | 256        |
| unet_rexnet13 | 32.14    | 9.34M |      | bilinear      | 256        |

