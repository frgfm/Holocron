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



## Training results

| Size (px) | Epochs | args                                                         | Top-1 accuracy | # Runs |
| --------- | ------ | ------------------------------------------------------------ | -------------- | ------ |
| 224       | 5      | imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 5 --opt radam --sched onecycle --loss label_smoothing | 66.88%         | 1      |
| 224       | 10     | imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 10 --opt radam --sched onecycle --loss label_smoothing | 76.18%         | 1      |
| 224       | 20     | imagenette2-320/ --model darknet53 --lr 5e-3 -b 32 -j 16 --epochs 20 --opt radam --sched onecycle --loss label_smoothing | 82.57%         | 1      |

