import argparse
import datetime
import glob
import math
import os
import random
import re
import shutil
from copy import deepcopy
from pathlib import Path
from time import sleep
import lpips
import lpips
import pyiqa
from PIL import Image
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from loguru import logger
import torch.nn.functional as F
import lpips
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from pyiqa.utils import imread2tensor
from tqdm import tqdm

from basicsr import DiffJPEG, USMSharp
from basicsr.data.degradations import random_add_poisson_noise_pt, random_add_gaussian_noise_pt
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from datapipe.datasets import create_dataset
from inference_resshift import get_configs, ResShiftSampler, get_parser

from trainer import replace_nan_in_batch
from utils import util_image, util_common, util_net

'''
    将多张numpy图片水平拼接在一起
    input: image list (numpy H x W x C - bgr)
    output: conbined_img (numpy H x W x C - bgr)
'''
def conbine_imgs(imgs):
    heights=[]
    widths=[]
    images=[]
    for img in imgs:
        h,w= img.shape[:2]
        widths.append(w)
        heights.append(h)
    max_height = max(heights)

    for i,img in enumerate(imgs):
        height, width = img.shape[:2]
        if height < max_height:
            img = cv2.resize(img,(int(width*max_height/height),max_height))
        images.append(img)
    combined = np.hstack(images)
    return combined
# 测试单个模型指标
def metrix_model(configs, sampler, lq_path, gt_path,img_key="plt", save_img=True, output_dir="result", out_logs=None):

    resshift_sampler = sampler

    mask_path = None

    # im_sr_tensors, im_srs = [0,1],[0,255] rgb
    shanghai_tz = datetime.timezone(datetime.timedelta(hours=+8))
    start_time = datetime.datetime.now(shanghai_tz)
    im_sr_tensors, im_srs = resshift_sampler.inference2(
        lq_path,
        output_dir,
        mask_path=mask_path,
        bs=args.bs,
        noise_repeat=False
    )
    end_time = datetime.datetime.now(shanghai_tz)
    print(f'完成采样...')
    im_srs = [util_image.bgr2rgb(im_sr) for im_sr in im_srs]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取 inference2 实际处理的文件顺序
    if os.path.isdir(lq_path):
        # 创建与 inference2 相同的数据集获取真实顺序
        data_config = {
            'type': 'base',
            'params': {
                'dir_path': str(lq_path),
                'transform_type': 'default',
                'transform_kwargs': {'mean': 0.5, 'std': 0.5},
                'need_path': True,
                'recursive': True,
                'length': None,
            }
        }
        dataset = create_dataset(data_config)
        real_order = [os.path.basename(data['path']) for data in dataset]
    else:
        real_order = [os.path.basename(lq_path)]


    im_lqs = []
    im_lq_tensors = []
    im_gts = []
    im_gt_tensors = []
    length = len(im_srs)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for filename in real_order:
        # 读取 LQ (保持与 inference2 相同的 RGB 处理)
        lq_file = os.path.join(lq_path, filename)
        im_lq = util_image.imread(lq_file, chn='rgb', dtype='uint8')
        im_lq_tensor = util_image.img2tensor(im_lq).to(device)
        im_lqs.append(im_lq)
        im_lq_tensors.append(im_lq_tensor)

        # 读取对应的 GT
        gt_file = os.path.join(gt_path, filename)  # 假设同名

        im_gt = util_image.imread(gt_file, chn='rgb', dtype='uint8')
        im_gt_tensor = util_image.img2tensor(im_gt,out_type=torch.float16).to(device)
        im_gts.append(im_gt)
        im_gt_tensors.append(im_gt_tensor)
    import lpips
    lpips_metric_vgg = lpips.LPIPS(net='vgg').to(device)
    lpips_ls = []
    psnr_ls = []
    ssim_ls = []
    musiq_ls=[]
    clip_iqa_ls=[]
    for i, im_sr_tensor in tqdm(enumerate(im_sr_tensors)):
        # 低清图像numpy
        im_lq = im_lqs[i]
        # 高清图像tensor
        im_gt_tensor = im_gt_tensors[i]
        # 高清图像numpy
        im_gt = im_gts[i]
        # 超分图像numpy
        im_sr = im_srs[i]

        # im_gt [0,255]] | im_gt_tensor [0,255] | im_sr [0,255] | im_sr_tensor [0,1]

        # input要求： [0,255]
        psnr = util_image.calculate_psnr(im_gt , im_sr ,ycbcr=configs.train.val_y_channel)
        psnr_ls.append(psnr)

        # input要求：[ [0,255]
        ssim = util_image.calculate_ssim(im_gt , im_sr,ycbcr=configs.train.val_y_channel)
        ssim_ls.append(ssim)
        with torch.no_grad():
            # input要求：[-1,1]
            LPIPS = lpips_metric_vgg(im_gt_tensor.squeeze(0) / 255 * 2 - 1, im_sr_tensor.squeeze(0) * 2 - 1).view(-1).item()
            lpips_ls.append(LPIPS)
            musiq = 0
            musiq_ls.append(musiq)
            ciqa =0
            clip_iqa_ls.append(ciqa)
        # combine_img = conbine_imgs([im_gt, im_sr])
        if save_img:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            ax1.imshow(im_lq)
            ax1.set_title("LQ")

            ax2.imshow(im_gt)
            ax2.set_title("GT")

            ax3.imshow(im_sr)
            ax3.set_title("SR")

            plt.suptitle(f'psnr:{psnr:.3f}  lpips:{LPIPS:.3f}  ssim:{ssim:.3f}  musiq:{musiq:.3f}  ciqa:{ciqa:.3f}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{i}-{img_key}.png")
            plt.close(fig)
    psnr = sum(psnr_ls) / length
    lpips = sum(lpips_ls) / length
    ssim = sum(ssim_ls) / length
    musiq = sum(musiq_ls) / length
    ciqa = sum(clip_iqa_ls) / length
    if out_logs is None:
        out_logs = f"metrix_logs/{img_key}-{gt_path}-psnr-{psnr:.4f}-lpips-{lpips:.4f}.log"
    logger.remove(0)
    logger.add(out_logs)
    logger.info(
        f"\nimg_key: {img_key}\n"
        f"lq path: {lq_path}\n"
        f"gt path: {gt_path}\n"
        f"number of images: {len(im_lqs)}\n"
        f"model: {configs.model.ckpt_path}\n"
        f"target: {configs.model.target}\n"
        f"lpips:{lpips:.4f}\n"
        f"psnr:{psnr:.4f}\n"
        f"ssim:{ssim:.4f}\n"
        f"musiq:{musiq:.4f}\n"
        f"ciqa:{ciqa:.4f}\n"
        f"start_time: {start_time.strftime('%Y-%m-%d %H:%M:%S') }\n"
        f"end_time: {end_time.strftime('%Y-%m-%d %H:%M:%S') }\n"
        f"times:{ end_time - start_time }\n")
    print(f'target:{configs.model.target}')
    print(f'model: {configs.model.ckpt_path}')
    print(f'lpips:{lpips:.4f}')
    print(f'psnr:{psnr:.4f}')
    print(f'ssim:{ssim:.4f}')
    print(f'musiq:{musiq:.4f}')
    print(f'ciqa:{ciqa:.4f}')
    print(f"times:{end_time - start_time}")
    return {
        "psnr": psnr_ls,
        "lpips": lpips_ls,
        "ssim": ssim_ls,
        "img_srs": im_srs,
        "img_lqs": im_lqs,
        "img_gt": im_gts,
    }

# 测试baseline模型指标
def metrix_model_baseline_model(configs, lq_path, gt_path, chop_stride,img_key=None, save_img=True, output_dir="result"):
    resshift_sampler = ResShiftSampler(
        configs,
        sf=args.scale,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_amp=True,
        seed=args.seed,
        padding_offset=configs.model.params.get('lq_size', 64),
    )
    key_split=configs.model.ckpt_path.split('\\')
    if len(key_split) == 1:
        key_split = configs.model.ckpt_path.split('/')
    img_key = img_key if img_key is not None else key_split[-1].split('.')[0]
    return metrix_model(configs, resshift_sampler, lq_path, gt_path, img_key, save_img, output_dir)



def flops(model,input=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.randint(
        0, 4,
        size=(1,),
        device=f"cuda:{0}",
    ).to(device)
    tensor = torch.rand(1, 3, 64, 64).to(device) if input is None else input
    lq = torch.rand(1, 3, 64, 64).to(device)
    # model(self._scale_input(z_t, t), t, **model_kwargs)
    flops = FlopCountAnalysis(model, (tensor,t,lq))
    total = flops.total()
    print(f"Total FLOPs: ", total)
    print(f"Total FLOPs (G): {total / 1e9:.4} G")  # 转换为十亿次单位(G)
    # 分析parameters
    # print(parameter_count_table(model))
    return total



def fid(generated_path,gt_path,device="cuda"):
    fid_metric = pyiqa.create_metric('fid', device=device)
    return fid_metric(generated_path,gt_path)

def generate_lq_gt_imgs(args,indir,outdir=None,num_imgs=None,g_gt=True):
    configs, chop_stride = get_configs(args)
    img_list = os.listdir(indir)
    random.seed(10000)
    random.shuffle(img_list)
    outdir = outdir if outdir is not None else indir
    gt_dir = Path(outdir) / 'gt'
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True)
    lq_dir = Path(outdir) / 'lq'
    if not lq_dir.exists():
        lq_dir.mkdir(parents=True)
    opts, opts_degradation = configs.data.train.params, configs.degradation
    opts['dir_paths'] = [indir, ]
    if num_imgs is not None:
        opts['length'] = num_imgs
    dataset = RealESRGANDataset(opts, mode='testing')
    for ii in range(len(dataset.paths)):
        data_dict1 = dataset.__getitem__(ii)
        prefix = 'realesrgan'
        data_dict2 = dataset.degrade_fun(
            opts_degradation,
            im_gt=data_dict1['gt'].unsqueeze(0),
            kernel1=data_dict1['kernel1'],
            kernel2=data_dict1['kernel2'],
            sinc_kernel=data_dict1['sinc_kernel'],
        )
        im_lq, im_gt = data_dict2['lq'], data_dict2['gt']
        im_lq, im_gt = util_image.tensor2img([im_lq, im_gt], rgb2bgr=True, min_max=(0, 1))  # uint8

        im_name = Path(data_dict1['gt_path']).stem
        im_path_gt = gt_dir / f'{im_name}.png'
        if g_gt:
            util_image.imwrite(im_gt, im_path_gt, chn='bgr', dtype_in='uint8')

        im_path_lq = lq_dir / f'{im_name}.png'
        util_image.imwrite(im_lq, im_path_lq, chn='bgr', dtype_in='uint8')


def rename(directory,key):
    """
    替换指定目录下所有图片文件名中的key
    例如：abc_X4.png -> abc.png
    参数:
        directory (str): 要处理的目录路径
        key (str): 需要替换的字符串，例如: "_X4" -> ""
    """
    # 支持的图片文件扩展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在")
        return

    # 遍历目录中的所有文件
    renamed_count = 0
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 只处理文件，跳过目录
        if not os.path.isfile(file_path):
            continue

        # 检查文件是否为图片
        _, ext = os.path.splitext(filename)
        if ext.lower() not in image_extensions:
            continue

        # 检查文件名中是否包含key
        if key in filename:
            # 创建新文件名（移除所有key出现）
            new_filename = filename.replace(key, '')
            new_file_path = os.path.join(directory, new_filename)

            # 避免文件名冲突
            counter = 1
            while os.path.exists(new_file_path):
                name, ext = os.path.splitext(new_filename)
                new_file_path = os.path.join(directory, f"{name}_{counter}{ext}")
                counter += 1

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"重命名: {filename} -> {os.path.basename(new_file_path)}")
            renamed_count += 1

    print(f"\n完成！共重命名了 {renamed_count} 个文件。")

def get_model(configs,rank=0):
    params = configs.model.get('params', dict)
    model = util_common.get_obj_from_str(configs.model.target)(**params)
    model.cuda()
    if configs.model.ckpt_path is not None:
        ckpt_path = configs.model.ckpt_path
        if rank == 0:
            logger.info(f"Initializing model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        util_net.reload_model(model, ckpt)
    return model

def batch_metrix(gt_dir,sr_dir,device ="cuda",border=0,ycbcr=True,only_musiq_iqa=False):
    device = device
    ssim_total = []
    lpips_total = []
    psnr_total = []
    musiq_total = []
    clip_iqa_total = []
    map = {}
    files=os.listdir(sr_dir)
    file_len=len(files)
    lpips_metric_vgg = lpips.LPIPS(net='vgg').to(device)
    musiq_metric = pyiqa.create_metric('musiq',device=device)
    clip_iqa_metric = pyiqa.create_metric('clipiqa',device=device)
    total = 0
    for i,f_name in enumerate(files):
        if not only_musiq_iqa:
            gt_path =os.path.join(gt_dir, f_name)
            try:
                gt = util_image.imread(gt_path) # [0,1] rgb
            except:
                print(f'Error:{f_name} not in gt_dir,please make sure lq_img_name == gt_img_name')
            gt_tensor = util_image.img2tensor(gt).to(device) # [0,1]
        sr_path = os.path.join(sr_dir, f_name)
        fn,ext = os.path.splitext(f_name)
        if ext not in ['.jpg','.png','.jpeg']:
            continue
        sr=util_image.imread(sr_path)
        sr_tensor = util_image.img2tensor(sr).to(device)
        if not only_musiq_iqa:
            psnr = util_image.calculate_psnr(gt *255,sr *255,border=border,ycbcr=ycbcr)
            psnr_total.append(psnr)
            ssim = util_image.calculate_ssim(gt *255,sr *255,border=border,ycbcr=ycbcr)
            ssim_total.append(ssim)
        with torch.no_grad():
            if not only_musiq_iqa:
                LPIPS = lpips_metric_vgg(gt_tensor *2-1, sr_tensor *2-1).view(-1).item()
                lpips_total.append(LPIPS)
            musiq = musiq_metric(sr_path)
            musiq_total.append(musiq)
            clip = clip_iqa_metric(sr_path)
            clip_iqa_total.append(clip)
        if not only_musiq_iqa:
            map[f_name] = {
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "musiq": musiq_metric(sr_path),
                "clip": clip
            }
        else:
            map[f_name] = {
                "musiq": musiq_metric(sr_path),
                "clip": clip
            }
        print(f"{i+1}/{file_len}: {f_name}")
        total+=1
    print(f"总计成功处理 {total} 个文件...")
    return sum(psnr_total)/max(1,len(psnr_total)),sum(lpips_total)/max(1,len(lpips_total)),sum(ssim_total)/max(1,len(ssim_total)),sum(musiq_total)/max(1,len(musiq_total)),sum(clip_iqa_total)/max(1,len(clip_iqa_total))
def bicubic(lq_path,out_path,scale_factor=4):
    files = os.listdir(lq_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    length = len(files)
    for i,f_name in enumerate(files):
        img_path = os.path.join(lq_path, f_name)
        img = util_image.imread(img_path)
        img_tensor = util_image.img2tensor(img).cuda()
        img_bic_tensor = torch.nn.functional.interpolate(img_tensor, size=None, scale_factor=scale_factor, mode='bicubic', align_corners=None)
        im_sr = util_image.tensor2img(img_bic_tensor, rgb2bgr=True, min_max=(0.0, 1.0))
        im_path = os.path.join(out_path, f_name)
        util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
        print(f'{i+1}/{length}: {f_name}')




def organize_and_rename_images(in_dir):
    """
    整理图片文件到gt和lq目录并统一重命名

    参数:
    in_dir: 输入目录路径，包含xxx_HR.png和xxx_LR4.png文件
    """

    # 创建目标目录
    gt_dir = os.path.join(in_dir, 'gt')
    lq_dir = os.path.join(in_dir, 'lq')
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(lq_dir, exist_ok=True)

    # 获取目录中的所有文件
    all_files = os.listdir(in_dir)

    # 处理HR文件
    hr_pattern = re.compile(r'(.+)_HR\.png$', re.IGNORECASE)
    for file_name in all_files:
        hr_match = hr_pattern.match(file_name)
        if hr_match:
            base_name = hr_match.group(1)
            new_name = f"{base_name}.png"
            src_path = os.path.join(in_dir, file_name)
            dst_path = os.path.join(gt_dir, new_name)
            shutil.move(src_path, dst_path)
            print(f"Moved {file_name} to gt/{new_name}")

    # 处理LR4文件
    lr_pattern = re.compile(r'(.+)_LR4\.png$', re.IGNORECASE)
    for file_name in all_files:
        lr_match = lr_pattern.match(file_name)
        if lr_match:
            base_name = lr_match.group(1)
            new_name = f"{base_name}.png"
            src_path = os.path.join(in_dir, file_name)
            dst_path = os.path.join(lq_dir, new_name)
            shutil.move(src_path, dst_path)
            print(f"Moved {file_name} to lq/{new_name}")

    print("文件整理完成!")
def ensure_consistent_format(img_gt, img_restored):
    """
    确保GT和恢复图像格式一致
    对于Set14数据集，如果GT是灰度，恢复图像也转换为灰度
    如果GT是彩色，恢复图像保持彩色
    """
    # 检查GT图像通道数
    if len(img_gt.shape) == 2:
        # GT是灰度图像
        if len(img_restored.shape) == 3:
            # 恢复图像是彩色，转换为灰度
            img_restored = cv2.cvtColor(img_restored, cv2.COLOR_BGR2GRAY)
    else:
        # GT是彩色图像
        if len(img_restored.shape) == 2:
            # 恢复图像是灰度，转换为彩色
            img_restored = cv2.cvtColor(img_restored, cv2.COLOR_GRAY2BGR)
        elif img_restored.shape[2] == 4:
            # 恢复图像是RGBA，转换为RGB
            img_restored = cv2.cvtColor(img_restored, cv2.COLOR_RGBA2BGR)

    return img_gt, img_restored
import os.path as osp
if __name__ == '__main__':
    args = get_parser()
    args.cfg_path='configs/realsr_swinunet_realesrgan256_test.yaml'
    in_path_dir= "testdata/Val_SR"
    (configs, chop_stride) = get_configs(args)
    configs.model.target = 'models.unet.UNetModelSwinPyConv5'
    configs.model.ckpt_path = 'weights/2layer-stu-16000-multi_step.pth'
    # weights/exp/no-no-mfr11-60000.pth
    # 'weights/dfs2-aff-pc-60000.pth'
    # weights/01-aff-mfr_60000.pth
    # 'weights/dfs-aff-no-best.pth'
    # weights/resshift_realsrx4_s4_v3.pth
    # weights/exp/mybase-4-90000.pth
    # 1、计算flops
    # model = get_model(configs)
    # flops(model)

    # 2、适合于消融测试
    # metrix_model_baseline_model(configs, in_path_dir+"/lq",
    #                             in_path_dir+"/gt",
    #                             chop_stride=chop_stride,
    #                             output_dir="test_set5" ,
    #                             save_img=False,
    #                             )


    # 3、对比实验测试指标：跨sr目录测试指标psnr、lpips、ssim...
    # rename("../DASR-master/results/DASR2/visualization/RealSet80","_DASR2")
    sr_imgs_dir = "../SinSR-main/outputs/2layer-stu-24000-one_step-stage2/realsr"
    key="2layer-stu-24000-one_step-stage2-realsr"
    gt_dir = "testdata/RealSRx4/gt"
    #
    s = f"gt_dir: {gt_dir}\n"
    psnr, lpips, ssim, m, c = batch_metrix(gt_dir, sr_imgs_dir)
    s+=f"{key}[{sr_imgs_dir}] —— PSNR: {psnr:.4f},LPIPS: {lpips:.4f},SSIM: {ssim:.4f},Musiq: {m.item():.4f},ClipIQA: {c.item():.4f}\n"
    s+="="*50
    print(s)
    shanghai_tz = datetime.timezone(datetime.timedelta(hours=+8))
    start_time = datetime.datetime.now(shanghai_tz)
    logger.remove(0)
    logger.add(f"m/{start_time.strftime('%Y-%m-%d-%H-%M-%S')}_{key}.log")
    logger.info(s)





    # 4、生成线性插值图片
    # bicubic("testdata/RealSet80","result/bicubic/realset80",scale_factor=4)



