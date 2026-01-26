# AMFDiff
AMFDiff: Adaptive Multi-scale Fusion Diffusion Model for Image Super-Resolution

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
