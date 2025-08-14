import glob
import math
import os
import random
import shutil
from copy import deepcopy
from pathlib import Path
from loguru import logger
import torch.nn.functional as F
import lpips
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from basicsr import DiffJPEG, USMSharp
from basicsr.data.degradations import random_add_poisson_noise_pt, random_add_gaussian_noise_pt
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from basicsr.data.transforms import paired_random_crop
from basicsr.utils.img_process_util import filter2D
from datapipe.datasets import create_dataset
from inference_resshift import get_configs, ResShiftSampler, get_parser

from trainer import replace_nan_in_batch
from utils import util_image

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
def metrix_model(args, sampler, lq_path, gt_path, img_key="plt", save_img=True, output_dir="result", out_logs=None):
    args.in_path = lq_path
    configs, chop_stride = get_configs(args)
    resshift_sampler = sampler

    mask_path = None

    # im_sr_tensors, im_srs = [0,1],[0,255] rgb
    im_sr_tensors, im_srs = resshift_sampler.inference2(
        args.in_path,
        args.out_path,
        mask_path=mask_path,
        bs=args.bs,
        noise_repeat=False
    )
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

    for i, im_sr_tensor in enumerate(im_sr_tensors):
        # 低清图像numpy
        im_lq = im_lqs[i]
        # 高清图像tensor
        im_gt_tensor = im_gt_tensors[i]
        # 高清图像numpy
        im_gt = im_gts[i]
        # 超分图像numpy
        im_sr = im_srs[i]

        # im_gt [0,255]] | im_gt_tensor [0,255] | im_sr [0,255] | im_sr_tensor [0,1]

        # input要求：[-1,1]
        LPIPS = lpips_metric_vgg(im_gt_tensor / 255 * 2 -1 , im_sr_tensor * 2 -1 ).view(-1).item()
        lpips_ls.append(LPIPS)

        # input要求： [0,255]
        psnr = util_image.calculate_psnr(im_gt , im_sr )
        psnr_ls.append(psnr)

        # input要求：[ [0,255]
        ssim = util_image.calculate_ssim(im_gt , im_sr )
        ssim_ls.append(ssim)


        # combine_img = conbine_imgs([im_gt, im_sr])
        if save_img:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            ax1.imshow(im_lq)
            ax1.set_title("LQ")

            ax2.imshow(im_gt)
            ax2.set_title("GT")

            ax3.imshow(im_sr)
            ax3.set_title("SR")

            plt.suptitle(f'psnr:{psnr:.3f}  lpips:{LPIPS:.3f}  ssim:{ssim:.3f}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{i}-{img_key}.png")
            plt.close(fig)
    psnr = sum(psnr_ls) / length
    lpips = sum(lpips_ls) / length
    ssim = sum(ssim_ls) / length
    if out_logs is None:
        out_logs = f"metrix_logs/{img_key}-psnr-{psnr:.4f}-lpips-{lpips:.4f}.log"
    logger.remove(0)
    logger.add(out_logs)
    aux_model = configs.aux_model.ckpt_path if hasattr(configs, "aux_model") else ""
    aux_model_target = configs.aux_model.target if hasattr(configs, "aux_model") else ""
    logger.info(
        f"\nimg_key: {img_key}\n"
        f"lq path: {lq_path}\n"
        f"gt path: {gt_path}\n"
        f"model: {configs.model.ckpt_path}\n"
        f"target: {configs.model.target}\n"
        f"aux_model: {aux_model}\n"
        f"aux target: {aux_model_target}\n"
        f"lpips:{lpips:.4f}\n"
        f"psnr:{psnr:.4f}\n"
        f"ssim:{ssim:.4f}\n"
        f"")

    print(f'model: {configs.model.ckpt_path}')
    print(f'lpips:{lpips:.4f}')
    print(f'psnr:{psnr:.4f}')
    print(f'ssim:{ssim:.4f}')
    return {
        "psnr": psnr_ls,
        "lpips": lpips_ls,
        "ssim": ssim_ls,
        "img_srs": im_srs,
        "img_lqs": im_lqs,
        "img_gt": im_gts,
    }


# 测试baseline模型指标
def metrix_model_baseline_model(args, lq_path, gt_path, img_key=None, save_img=True, output_dir="result"):
    args.in_path = lq_path
    configs, chop_stride = get_configs(args)
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
    img_key = img_key if img_key is not None else configs.model.ckpt_path.split('\\')[-1].split('.')[0]
    return metrix_model(args, resshift_sampler, lq_path, gt_path, img_key, save_img, output_dir)

# 批量测试主模型，可以保存合并超分图片
def batch_metrix_model(args, cfg_paths, lq_path, gt_path, img_key="plt", save_img=True, output_dir="result"):
    img_srss = []
    img_lqs = None
    img_gts = None
    for ii, cfg_path in enumerate(cfg_paths):
        args.cfg_path = cfg_path
        result = metrix_model_baseline_model(args, lq_path, gt_path, img_key, save_img=save_img, output_dir=output_dir)
        if img_lqs is None:
            img_lqs = result["img_lqs"]
        if img_gts is None:
            img_gts = result["img_gts"]
        img_srss.append(result['img_srs'])
    for ii, img_lq in enumerate(img_lqs):
        imgs = [img_lq, img_gts[ii]]
        for jj, img_srs in enumerate(img_srss):
            imgs.append(img_srss[jj][ii])
        img = conbine_imgs(imgs)
        if save_img:
            cv2.imwrite(f"{output_dir}/{img_key}_conbined_{ii}.png", img)


def remove_imgs(input_path, key):
    """
       删除指定目录及其子目录中所有以指定字符串开头的PNG图片

       参数:
       input_path (str): 要搜索的目标目录路径
       key (str): 要匹配的文件名前缀
       """
    # 检查输入路径是否存在
    if not os.path.exists(input_path):
        print(f"错误: 路径 '{input_path}' 不存在")
        return

    # 遍历目录及其所有子目录
    for root, _, files in os.walk(input_path):
        # 查找所有以key结尾的PNG文件（不区分大小写）
        for filename in files:
            if filename.lower().endswith(key + '.png'):
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print(f"已删除: {filepath}")
                except Exception as e:
                    print(f"删除失败 {filepath}: {str(e)}")

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


if __name__ == '__main__':
    args = get_parser()
    args.cfg_path='configs/realsr_swinunet_realesrgan256_test.yaml'
    in_path_dir= "testdata/Val_SR"
    metrix_model_baseline_model(args, in_path_dir+"/lq",
                                in_path_dir+"/gt",
                                output_dir="result_SR" ,
                                save_img=False,
                                )

    # remove_imgs("result","aux_model_diff_new_10000")
    # generate_lq_gt_imgs(args, "testdata/RealSet80",outdir='testdata/Val_RealSet80')