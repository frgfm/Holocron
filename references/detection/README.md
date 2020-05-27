# Object detection

The sample training script was made to train object detection models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip installe -e Holocron/. --upgrade
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection) for you! From there, you can run your training with the following command

```bash
python train.py VOC2012 --model yolov2 --lr 1e-5 -b 32 -j 16 --epochs 20 --opt radam --sched onecycle
```



## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset. Since the mAP does not allow easy interpretation by humans, the selected performance metrics are the recall and precision at IoU 0.5. A prediction is considered as correct if:

- it is the best acceptable localization candidate (highest IoU among predictions with the GT, and IoU >= 0.5)
- the top predicted probabilities is for the class label of the matched ground truth object.

Here, the recall being the ratio of correctly predicted ground truth predictions by the total number of ground truth objects, and the precision being the ratio of correctly predicted ground truth predictions by the total number of predicted boxes.

| Size (px) | Epochs | args                                                         | Recall@.5 | Precision@.5 | # Runs |
| --------- | ------ | ------------------------------------------------------------ | --------- | ------------ | ------ |
| 416       | 80     | VOC2012 --model yolov2 --lr 1e-4 -b 32 -j 16 --epochs 80 --opt radam --sched onecycle | 13.82%    | 2.56%        | 1      |



## Model zoo

| Model  | Recall@.5 | Precision@.5 | Param # | MACs | Interpolation | Image size |
| ------ | --------- | ------------ | ------- | ---- | ------------- | ---------- |
| yolov2 | 13.82%    | 2.56%        | 67.14M  |      | bilinear      | 416        |

