# Object detection

The sample training script was made to train object detection models on [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip install -e Holocron/. --upgrade
```

No need to download the dataset, torchvision will handle [this](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection) for you! From there, you can run your training with the following command

```bash
python train.py VOC2012 --arch yolov2 --lr 1e-5 -b 32 -j 16 --epochs 20 --opt radam --sched onecycle
```



## Personal leaderboard

### PASCAL VOC 2012

Performances are evaluated on the validation set of the dataset. Since the mAP does not allow easy interpretation by humans, the performance metrics have been changed here. 

A prediction is considered as correct if it checks two criteria:

- Localization: it is the best acceptable localization candidate (highest IoU among predictions with the GT, and IoU >= 0.5)
- Classification: the top predicted probabilities is for the class label of the matched ground truth object.

Then we define:

- **Localization error rate**: with loc_recall being the matching rate of ground truth boxes, and loc_precision being the matching rate of predicted boxes, we define the localization error as 1 - (harmonic mean of localization loc_recall & loc_precision)
- **Classification error rate**: classification error rate of matched predictions.
- **Detection error rate**: with det_recall being the correctness rate of ground truth boxes, and det_precision being the correctness rate of predicted boxes, we define the localization error as 1 - (harmonic mean of localization det_recall & det_precision)

Here, the recall being the ratio of correctly predicted ground truth predictions by the total number of ground truth objects, and the precision being the ratio of correctly predicted ground truth predictions by the total number of predicted boxes.

| Size (px) | Epochs | args                                                         | Loc@.5 | Clf@.5 | Det@.5 | # Runs |
| --------- | ------ | ------------------------------------------------------------ | ------ | ------ | ------ | ------ |
| 416       | 40     | VOC2012 --arch yolov2 --img-size 416 --lr 5e-4 -b 64 -j 16 --epochs 40 --opt tadam --freeze-backbone --sched onecycle | 83.09  | 52.82  | 92.02  | 1      |



## Model zoo

| Model  | Loc@.5 | Clf@.5 | Det@.5 | Param # | MACs | Interpolation | Image size |
| ------ | ------ | ------ | ------ | ------- | ---- | ------------- | ---------- |
| yolov2 | 83.09  | 52.82  | 92.02  | 50.65M  |      | bilinear      | 416        |

