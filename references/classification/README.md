# Image classification

Since I do not own enough computing power to iterate over ImageNet full training, this section involves training on a subset of ImageNet, called [Imagenette](https://github.com/fastai/imagenette).

## Getting started

Ensure that you have holocron installed

```bash
git clone https://github.com/frgfm/Holocron.git
pip installe -e Holocron/. --upgrade
```

Download [Imagenette](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz) and extract it where you want

```bash
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xvzf imagenette2-320.tgz
```

From there, you can run your training with the following command

```
python train.py imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 40 --opt radam --sched onecycle --loss label_smoothing
```



## Personal leaderboard

### Imagenette

| Size (px) | Epochs | args                                                         | Top-1 accuracy | # Runs |
| --------- | ------ | ------------------------------------------------------------ | -------------- | ------ |
| 224       | 5      | imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 5 --opt radam --sched onecycle --loss label_smoothing | 66.88%         | 1      |
| 224       | 10     | imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 10 --opt radam --sched onecycle --loss label_smoothing | 76.18%         | 1      |
| 224       | 20     | imagenette2-320/ --model darknet19 --lr 5e-4 -b 32 -j 16 --epochs 20 --opt radam --sched onecycle --loss label_smoothing | 84.43%         | 1      |
| 224       | 40     | imagenette2-320/ --model darknet19 --lr 5e-4 -b 32 -j 16 --epochs 40 --opt radam --sched onecycle --loss label_smoothing | 90.47%         | 1      |



## Model zoo

| Model     | Accuracy@1 (Err) | Param # | MACs  | Interpolation | Image size |
| --------- | ---------------- | ------- | ----- | ------------- | ---------- |
| darknet53 | 87.62 (12.38)    | 40.60M  | 7.13G | bilinear      | 224        |
| darknet19 | 90.47 (9.53)     | 19.83M  | 2.71G | bilinear      | 224        |
| darnet24  | 85.48 (14.52)    | 22.40M  | 4.21G | bilinear      | 224        |

