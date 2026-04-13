import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse


def find_subimage_position(large_image_path, subimage_path, method='cv2.TM_CCOEFF_NORMED'):
    """
    在大型图像中查找子图像的位置

    参数:
    - large_image_path: 大型图像路径
    - subimage_path: 子图像路径
    - method: 匹配方法，可选:
        cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED

    返回:
    - top_left: 子图像左上角坐标 (x, y)
    - bottom_right: 子图像右下角坐标 (x, y)
    - match_val: 匹配度得分
    """

    # 读取图像
    large_img = cv2.imread(large_image_path)
    sub_img = cv2.imread(subimage_path)

    if large_img is None or sub_img is None:
        print("错误: 无法读取图像文件")
        return None, None, None

    # 转换为灰度图进行匹配
    large_gray = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    sub_gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)

    # 获取子图像尺寸
    h, w = sub_gray.shape

    # 执行模板匹配
    if method == 'cv2.TM_CCOEFF_NORMED':
        method = cv2.TM_CCOEFF_NORMED
    elif method == 'cv2.TM_CCOEFF':
        method = cv2.TM_CCOEFF
    elif method == 'cv2.TM_CCORR_NORMED':
        method = cv2.TM_CCORR_NORMED
    elif method == 'cv2.TM_SQDIFF_NORMED':
        method = cv2.TM_SQDIFF_NORMED
    else:
        method = cv2.TM_CCOEFF_NORMED  # 默认方法

    result = cv2.matchTemplate(large_gray, sub_gray, method)

    # 获取最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 对于SQDIFF方法，取最小值位置
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
        match_val = min_val
    else:
        top_left = max_loc
        match_val = max_val

    bottom_right = (top_left[0] + w, top_left[1] + h)

    return top_left, bottom_right, match_val, large_img, sub_img


def visualize_match(large_img, sub_img, top_left, bottom_right, match_val):
    """可视化匹配结果"""

    # 创建可视化图像
    vis_img = large_img.copy()

    # 在大型图像上绘制矩形框
    cv2.rectangle(vis_img, top_left, bottom_right, (0, 255, 0), 3)

    # 设置绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示原始大图
    axes[0].imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始大图')
    axes[0].axis('off')

    # 显示匹配结果
    axes[1].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'匹配位置 (匹配度: {match_val:.4f})')
    axes[1].axis('off')

    # 显示子图像
    axes[2].imshow(cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('子图像')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    return fig


def extract_crop_coordinates(top_left, bottom_right, sub_img_shape, padding=0):
    """
    提取裁剪坐标，可选择添加padding

    参数:
    - top_left: 左上角坐标 (x, y)
    - bottom_right: 右下角坐标 (x, y)
    - sub_img_shape: 子图像形状 (height, width)
    - padding: 额外填充像素数

    返回:
    - crop_coords: 裁剪坐标字典
    """

    h, w = sub_img_shape[:2]

    # 计算带padding的坐标
    x1 = max(0, top_left[0] - padding)
    y1 = max(0, top_left[1] - padding)
    x2 = min(bottom_right[0] + padding, bottom_right[0] + w)  # 限制不超过大图边界
    y2 = min(bottom_right[1] + padding, bottom_right[1] + h)

    crop_coords = {
        'top_left': (x1, y1),
        'bottom_right': (x2, y2),
        'width': x2 - x1,
        'height': y2 - y1,
        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
        'original_size': (w, h),
        'padding': padding
    }

    return crop_coords


def apply_crop_to_all_images(image_paths, crop_coords, output_dir='cropped_results'):
    """
    将相同的裁剪框应用到所有图像

    参数:
    - image_paths: 图像路径列表
    - crop_coords: 裁剪坐标字典
    - output_dir: 输出目录
    """

    import os

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}
    top_left = crop_coords['top_left']
    bottom_right = crop_coords['bottom_right']

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is not None:
            # 执行裁剪
            cropped = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 保存结果
            output_path = os.path.join(output_dir, f'crop_{img_name}')
            cv2.imwrite(output_path, cropped)

            results[img_name] = {
                'path': output_path,
                'size': cropped.shape
            }
            print(f"已裁剪: {img_name} -> {output_path} (大小: {cropped.shape})")
        else:
            print(f"警告: 无法读取 {img_path}")

    return results


def main():


    ori_path= "picture/realset80/Canon_004_LR4-yes/crop/all3-2layer-55000_crop.png"
    patch_path= "picture/realset80/Canon_004_LR4-yes/crop2/all3-2layer-55000_crop_crop.png"
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='从局部放大图反向定位裁剪区域')
    parser.add_argument('--large', default=None,help='大型图像路径')
    parser.add_argument('--crop',default=None,  help='局部放大图像路径')
    parser.add_argument('--method', default='cv2.TM_CCOEFF_NORMED',
                        help='匹配方法 (cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED等)')
    parser.add_argument('--padding', type=int, default=0, help='额外填充像素')
    parser.add_argument('--apply_to', nargs='+', help='应用相同裁剪的其他图像路径')

    args = parser.parse_args()
    args.large=ori_path if args.large is None else args.large
    args.crop=patch_path if args.crop is None else args.crop
    # 1. 查找子图像位置
    print(f"在大型图像中查找子图像...")
    top_left, bottom_right, match_val, large_img, sub_img = find_subimage_position(
        args.large, args.crop, args.method
    )

    if top_left is None:
        print("无法找到匹配位置")
        return

    print(f"\n匹配结果:")
    print(f"  左上角: {top_left}")
    print(f"  右下角: {bottom_right}")
    print(f"  匹配度: {match_val:.6f}")
    print(f"  子图像大小: {sub_img.shape[:2]}")

    # 2. 可视化匹配结果
    print(f"\n生成可视化结果...")
    visualize_match(large_img, sub_img, top_left, bottom_right, match_val)

    # 3. 提取裁剪坐标
    crop_coords = extract_crop_coordinates(
        top_left, bottom_right, sub_img.shape, args.padding
    )

    print(f"\n裁剪坐标:")
    for key, value in crop_coords.items():
        if key not in ['top_left', 'bottom_right', 'center']:
            print(f"  {key}: {value}")

    print(f"  左上角: {crop_coords['top_left']}")
    print(f"  右下角: {crop_coords['bottom_right']}")
    print(f"  中心点: {crop_coords['center']}")

    # 4. 如果提供了其他图像，应用相同裁剪
    if args.apply_to:
        print(f"\n将相同裁剪应用到其他图像...")
        apply_crop_to_all_images(args.apply_to, crop_coords)


if __name__ == "__main__":
    main()