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

The results are test on minival set.

Trained models can be download in [Google Drive](https://drive.google.com/drive/folders/1EWtLhWSGuJVtMCS8mTvKNxdYYpz7ufjV?usp=sharing).

| Backbone  | Style   | GN  | MS train | Lr schd |  GPUs | Inf time (fps) | mask AP 
|:---------:|:-------:|:----:|:-------:|:-------:|:-----:|:--------------:|:------:|
| R-50      | caffe   | Y    | N       | 1x      |  4    | 8.9/23.9       | 28.9   |
| R-101     | caffe   | Y    | N       | 1x      |  4    | -              | 30.7   | 
| X-101     | pytorch   | Y  | N       | 1x      |  4    | -              | 32.5   | 

| Backbone  | Style   | GN  | MS train | Lr schd |  GPUs | Inf time (fps) | mask AP|
|:---------:|:-------:|:----:|:-------:|:-------:|:-----:|:--------------:|:------:|
| R-50      | caffe   | Y    | N       | 1x      |  32    | 8.9/23.9      | 29.1   | 
| R-101     | caffe   | Y    | N       | 1x      |  32    | -             | 30.4   |
| X-101     | pytorch | Y    | N       | 1x      |  32    | -             | 32.6   | 
| R-50      | caffe   | Y    | Y       | 2x      |  32    | 8.9/23.9      | 30.5   | 
| R-101     | caffe   | Y    | Y       | 2x      |  32    | -             | 31.9   |
| X-101     | pytorch | Y    | Y       | 2x      |  32    | -             | 33.5   | 
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


