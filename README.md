# AMFDiff
AMFDiff: Adaptive Multi-scale Fusion Diffusion Model for Image Super-Resolution

## Requirements
* Python 3.10, Pytorch 2.1.2, [xformers](https://github.com/facebookresearch/xformers) 0.0.23
* More detail (See [environment.yml](environment.yml))
A suitable [conda](https://conda.io/) environment named `resshift` can be created and activated with:

```
conda create -n amfdiff python=3.10
conda activate amfdiff
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
conda activate amfdiff
```

### Training
#### :turtle: Preparing stage
1. Download the pre-trained VQGAN model from this [link](https://github.com/zsyOAOA/ResShift/releases) and put it in the folder of 'weights'
2. Adjust the data path in the [config](configs) file. 
3. Adjust batchsize according your GPUS. 
    + configs.train.batch: [training batchsize, validation batchsize] 
    + configs.train.microbatch: total batchsize = microbatch * #GPUS * num_grad_accumulation


#### :whale: Real-world Image Super-resolution 
```
python main.py --cfg_path configs/py-realsr.yaml --save_dir logs 
```

## Fast Testing
#### :tiger: Real-world image super-resolution

```
python inference_resshift.py -i [image folder/image path] -o [result folder] --task realsr --scale 4 --version v3
python inference_resshift.py -i testdata/RealSet65/0014.jpg -o result --task realsr --scale 4 --version v3
python inference_resshift.py" -i testdata/RealSet65/0014.jpg -o result --task realsr --scale 4 --version v3 --cfg_path configs/realsr_swinunet_realesrgan256_my.yaml
```

## Online Demo
You can try our method through an online demo:
```
python app.py
```

## License

This project is licensed under <a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.

## Acknowledgement

This project is based on [Improved Diffusion Model](https://github.com/openai/improved-diffusion), [LDM](https://github.com/CompVis/latent-diffusion), and [BasicSR](https://github.com/XPixelGroup/BasicSR). We also adopt [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) to synthesize the training data for real-world super-resolution. Thanks for their awesome works.

