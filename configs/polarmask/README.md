# PolarMask: Single Shot Instance Segmentation with Polar Representation

## Introduction

```
@article{xie2019polarmask,
  title={PolarMask: Single Shot Instance Segmentation with Polar Representation},
  author={Xie, Enze and Sun, Peize and Song, Xiaoge and Wang, Wenhai and Liu, Xuebo and Liang, Ding and Shen, Chunhua and Luo, Ping},
  journal={arXiv preprint arXiv:1909.13226},
  year={2019}
}
```

## Results and Models
For 32 gpus, we set 2000 iters to warmup instead of 500. So the total epoches is 14 for 1x.    
And the performance is similar to 4gpus. Most of experiments are run on 32gpus in paper to fasten the process.

| Backbone  | Style   | GN  | MS train | Lr schd |  GPUs | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:----:|:-------:|:-------:|:-----:|:--------------:|:------:|:--------:|
| R-50      | caffe   | Y    | N       | 1x      |  4    | 23.9           | 28.9   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth) |
| R-101     | caffe   | Y    | N       | 1x      |  4    | -              | 30.7   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20190516-42e6f62d.pth) |
| X-101     | caffe   | Y    | N       | 1x      |  4    | -              | 32.5   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_20190516-a36c0872.pth) |

| Backbone  | Style   | GN  | MS train | Lr schd |  GPUs | Inf time (fps) | mask AP | Download |
|:---------:|:-------:|:----:|:-------:|:-------:|:-----:|:--------------:|:------:|:--------:|
| R-50      | caffe   | Y    | N       | 1x      |  32    | 23.9           | 29.1   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r50_caffe_fpn_gn_2x_4gpu_20190516-f7329d80.pth) |
| R-101     | caffe   | Y    | N       | 1x      |  32    | -              | 30.4   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20190516-42e6f62d.pth) |
| X-101     | caffe   | Y    | N       | 1x      |  32    | -              | 32.6   | [model](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_mstrain_640_800_x101_64x4d_fpn_gn_2x_20190516-a36c0872.pth) |

**Notes:**
- The X-101 backbone is X-101-64x4d.
- Dataloader is rewrited and it is slow because generating labels for rays is complex. We will try to speed up it in the futher.
- All models are trained with 1x and without data augmentation. We will release 2x with ms train model in the future.



**Train:**
##### 1. 4gpu train
- ```sh ./tools/dist_train.sh  configs/polarmask/4gpu/polar_768_1x_r50.py 4 --launcher pytorch --work_dir ./work_dirs/polar_768_1x_r50_4gpu```

##### 2. 32gpu train
- ```srun -p VI_ID_TITANX --job-name=PolarMask --gres=gpu:4 --ntasks=32 --ntasks-per-node=4 --kill-on-bad-exit=1 python -u tools/train.py configs/polarmask/32gpu/polar_768_1x_r50.py --launcher=slurm --work_dir ./work_dirs/polar_768_1x_r50_32gpu```



**Test:**
##### 4gpu test
- ```sh tools/dist_test.sh configs/polarmask/4gpu/polar_768_1x_r50.py ./work_dirs/polar_768_1x_r50_4gpu/latest.pth 4 --out work_dirs/trash/res.pkl --eval segm```


