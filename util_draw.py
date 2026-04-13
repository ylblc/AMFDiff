import math
import os
import random
import shutil
from pathlib import Path

import torch
from PIL import Image, ImageFont, ImageDraw
from typing import Dict, List, Tuple, Optional, Union

from matplotlib import pyplot as plt
import seaborn as sns

from utils import util_image


def create_sr_comparison_figure(
        image_paths: Dict[str, str],
        output_path: str = "super_resolution_comparison.png",
        layout: Tuple[int, int] = (2, 2),  # (rows, columns) for SR images
        labels: Optional[Dict[str, str]] = None,
        font_path: str = "arial.ttf",
        font_size: int = 20,
        dpi: Tuple[int, int] = (300, 300),
        background_color: str = 'white',
        text_color: str = 'black',
        text_bg_color: Optional[str] = None,  # 文字背景颜色
        h_spacing: int = 20,  # 水平间隔
        v_spacing: int = 20,  # 垂直间隔
        text_area_height: int = 40,  # 文字区域高度
        resize_to_gt: bool = True,
        border_width: int = 0,  # 图片边框宽度
        border_color: str = 'black',  # 图片边框颜色
        gt_scale: float = 1.5  # GT图放大比例
) -> None:
    """
    创建超分对比图的通用函数（GT图占据两行两列并放大）

    参数:
    image_paths: 图像路径字典，必须包含'gt'键，其他键为模型名称
    output_path: 输出图像路径
    layout: SR图像的布局(行数, 列数)
    labels: 图像标签字典，键与image_paths相同
    font_path: 字体路径
    font_size: 字体大小
    dpi: 输出图像DPI
    background_color: 背景颜色
    text_color: 文字颜色
    text_bg_color: 文字背景颜色
    h_spacing: 水平间隔
    v_spacing: 垂直间隔
    text_area_height: 文字区域高度
    resize_to_gt: 是否将所有SR图像调整为与GT相同尺寸
    border_width: 图片边框宽度
    border_color: 图片边框颜色
    gt_scale: GT图放大比例
    """

    # 打开所有图像
    images = {name: Image.open(path) for name, path in image_paths.items()}

    # 确保包含GT图像
    if 'gt' not in images:
        raise ValueError("图像路径中必须包含'gt'键")

    # 获取GT图像尺寸并放大
    gt_img = images['gt']
    gt_width, gt_height = gt_img.size
    scaled_gt_width = int(gt_width * gt_scale)
    scaled_gt_height = int(gt_height * gt_scale)
    scaled_gt_img = gt_img.resize((scaled_gt_width, scaled_gt_height), Image.Resampling.LANCZOS)

    # 调整SR图像尺寸（如果需要）
    if resize_to_gt:
        for name in images:
            if name != 'gt':
                images[name] = images[name].resize((gt_width, gt_height), Image.Resampling.LANCZOS)

    # 计算单张SR图片的总高度（图像高度 + 文字区域高度）
    single_img_height = gt_height + text_area_height

    # 计算GT图片的总高度（放大后的图像高度 + 文字区域高度）
    gt_total_height = scaled_gt_height + text_area_height

    # 计算画布尺寸
    sr_rows, sr_cols = layout

    # GT占据两行两列，所以SR图像从第三列开始
    canvas_width = scaled_gt_width + h_spacing + sr_cols * (gt_width + h_spacing)

    # 画布高度 = GT区域高度 和 SR区域高度的最大值
    sr_total_height = sr_rows * single_img_height + (sr_rows - 1) * v_spacing
    canvas_height = max(gt_total_height, sr_total_height)

    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(canvas)

    # 尝试加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # 放置GT图像（左上角，占据两行两列）
    gt_x = 0
    gt_y = 0

    # 添加边框（如果需要）
    if border_width > 0:
        draw.rectangle(
            [gt_x - border_width, gt_y - border_width,
             gt_x + scaled_gt_width + border_width, gt_y + scaled_gt_height + border_width],
            outline=border_color, width=border_width
        )

    canvas.paste(scaled_gt_img, (gt_x, gt_y))

    # 添加GT文字标注（底部）
    gt_label = labels.get('gt', 'Ground Truth') if labels else 'Ground Truth'
    text_bbox = draw.textbbox((0, 0), gt_label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = gt_x + (scaled_gt_width - text_width) // 2
    text_y = gt_y + scaled_gt_height + (text_area_height - font_size) // 2

    # 添加文字背景（如果需要）
    if text_bg_color:
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + font_size + padding],
            fill=text_bg_color
        )

    draw.text((text_x, text_y), gt_label, font=font, fill=text_color)

    # 放置SR图像（从第三列开始）
    sr_images = [name for name in images if name != 'gt']

    for idx, name in enumerate(sr_images):
        if idx >= sr_rows * sr_cols:
            break  # 防止图像数量超过布局容量

        row = idx // sr_cols
        col = idx % sr_cols

        # 计算SR图像位置（从第三列开始）
        x = scaled_gt_width + h_spacing + col * (gt_width + h_spacing)
        y = row * (single_img_height + v_spacing)

        # 添加边框（如果需要）
        if border_width > 0:
            draw.rectangle(
                [x - border_width, y - border_width,
                 x + gt_width + border_width, y + gt_height + border_width],
                outline=border_color, width=border_width
            )

        canvas.paste(images[name], (x, y))

        # 添加SR文字标注（底部）
        label = labels.get(name, name) if labels else name
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x_pos = x + (gt_width - text_width) // 2
        text_y_pos = y + gt_height + (text_area_height - font_size) // 2

        # 添加文字背景（如果需要）
        if text_bg_color:
            padding = 5
            draw.rectangle(
                [text_x_pos - padding, text_y_pos - padding,
                 text_x_pos + text_width + padding, text_y_pos + font_size + padding],
                fill=text_bg_color
            )

        draw.text((text_x_pos, text_y_pos), label, font=font, fill=text_color)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # 保存结果
    canvas.save(output_path, dpi=dpi)
    print(f"对比图已保存至: {output_path}")


def create_local_zoom_comparison(
        gt_image_path: str,
        sr_image_paths: Dict[str, str],
        zoom_region: Tuple[int, int, int, int],  # (x, y, width, height)
        output_path: str = "local_zoom_comparison.png",
        layout: Tuple[int, int] = (2, 2),  # (rows, columns) for SR zoom images
        labels: Optional[Dict[str, str]] = None,
        rect_color: str = "red",
        rect_width: int = 3,
        zoom_scale: float = 2.0,
        font_path: str = "arial.ttf",
        font_size: int = 16,
        dpi: Tuple[int, int] = (300, 300),
        background_color: str = 'white',
        text_color: str = 'white',
        text_outline: str = 'black'
) -> None:
    """
    创建局部放大对比图的通用函数

    参数:
    gt_image_path: GT图像路径
    sr_image_paths: 超分图像路径字典，键为模型名称，值为图像路径
    zoom_region: 放大区域 (x, y, width, height)
    output_path: 输出图像路径
    layout: SR放大图像的布局(行数, 列数)
    labels: 图像标签字典
    rect_color: 矩形框颜色
    rect_width: 矩形框线宽
    zoom_scale: 局部放大倍数
    font_path: 字体路径
    font_size: 字体大小
    dpi: 输出图像DPI
    background_color: 背景颜色
    text_color: 文字颜色
    text_outline: 文字描边颜色
    """

    # 打开GT图像
    gt_img = Image.open(gt_image_path)
    gt_width, gt_height = gt_img.size

    # 在GT图像上绘制红色矩形框
    gt_with_rect = gt_img.copy()
    draw = ImageDraw.Draw(gt_with_rect)
    x, y, w, h = zoom_region
    draw.rectangle([x, y, x + w, y + h], outline=rect_color, width=rect_width)

    # 从GT图像裁剪放大区域
    gt_zoom = gt_img.crop((x, y, x + w, y + h))
    gt_zoom = gt_zoom.resize((int(w * zoom_scale), int(h * zoom_scale)), Image.Resampling.LANCZOS)

    # 打开并处理所有SR图像
    sr_images = {}
    sr_zooms = {}

    for name, path in sr_image_paths.items():
        sr_img = Image.open(path)
        sr_images[name] = sr_img

        # 计算SR图像上的对应区域（考虑可能的尺寸差异）
        sr_width, sr_height = sr_img.size
        scale_x = sr_width / gt_width
        scale_y = sr_height / gt_height

        sr_x = int(x * scale_x)
        sr_y = int(y * scale_y)
        sr_w = int(w * scale_x)
        sr_h = int(h * scale_y)

        # 裁剪并放大
        sr_zoom = sr_img.crop((sr_x, sr_y, sr_x + sr_w, sr_y + sr_h))
        sr_zoom = sr_zoom.resize((int(w * zoom_scale), int(h * zoom_scale)), Image.Resampling.LANCZOS)
        sr_zooms[name] = sr_zoom

    # 计算画布尺寸
    zoom_width, zoom_height = gt_zoom.size
    sr_rows, sr_cols = layout

    # 左侧: GT图像 + 红色矩形框
    # 右侧: 上方是GT放大区域，下方是SR放大区域
    canvas_width = gt_width + max(zoom_width * sr_cols, gt_width // 2)
    canvas_height = max(gt_height, zoom_height * (1 + sr_rows))

    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=background_color)

    # 放置GT图像（带红色矩形框）
    canvas.paste(gt_with_rect, (0, 0))

    # 放置GT放大区域（右侧顶部）
    gt_zoom_x = gt_width + 10
    gt_zoom_y = 10
    canvas.paste(gt_zoom, (gt_zoom_x, gt_zoom_y))

    # 放置SR放大区域
    sr_images_list = list(sr_zooms.items())
    for idx, (name, zoom_img) in enumerate(sr_images_list):
        if idx >= sr_rows * sr_cols:
            break

        row = idx // sr_cols
        col = idx % sr_cols

        x_pos = gt_zoom_x + col * zoom_width
        y_pos = gt_zoom_y + zoom_height + 10 + row * zoom_height

        canvas.paste(zoom_img, (x_pos, y_pos))

    # 添加标签
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)

    # GT图像标签
    gt_label = labels.get('gt', 'Ground Truth') if labels else 'Ground Truth'
    draw.text((10, 10), gt_label, font=font, fill=text_color)

    # GT放大区域标签
    gt_zoom_label = labels.get('gt_zoom', 'GT Zoom') if labels else 'GT Zoom'
    # 添加文字描边
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((gt_zoom_x + dx, gt_zoom_y + dy), gt_zoom_label, font=font, fill=text_outline)
    draw.text((gt_zoom_x, gt_zoom_y), gt_zoom_label, font=font, fill=text_color)

    # SR放大区域标签
    for idx, (name, _) in enumerate(sr_images_list):
        if idx >= sr_rows * sr_cols:
            break

        row = idx // sr_cols
        col = idx % sr_cols

        x_pos = gt_zoom_x + col * zoom_width
        y_pos = gt_zoom_y + zoom_height + 10 + row * zoom_height

        label = labels.get(name, name) if labels else name

        # 添加文字描边
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            draw.text((x_pos + dx, y_pos + dy), label, font=font, fill=text_outline)
        draw.text((x_pos, y_pos), label, font=font, fill=text_color)

    # 保存结果
    canvas.save(output_path, dpi=dpi)
    print(f"局部放大对比图已保存至: {output_path}")


def crop_images2(
        input_paths: Union[List[str], str],
        output_dir: str,
        target_size: Tuple[int, int],
        crop_mode: str = "center",
        position: Optional[Tuple[int, int]] = None,
        allow_upscaling: bool = False,
        suffix: str = "_cropped",
        format: str = "auto",
        scale=None
) -> List[str]:
    """
    通用图像裁剪函数，可以将一组照片裁剪成相同的大小

    参数:
    input_paths: 输入图像路径列表或目录路径
    output_dir: 输出目录
    target_size: 目标尺寸 (width, height)
    crop_mode: 裁剪模式，可选值:
        - "center": 中心裁剪 (默认)
        - "random": 随机位置裁剪
        - "position": 指定位置裁剪 (需要提供position参数)
        - "top-left": 左上角裁剪
        - "top-right": 右上角裁剪
        - "bottom-left": 左下角裁剪
        - "bottom-right": 右下角裁剪
    position: 当crop_mode为"position"时，指定裁剪的左上角坐标 (x, y)
    allow_upscaling: 是否允许放大图像（如果原图小于目标尺寸）
    suffix: 输出文件名后缀
    format: 输出格式，"auto"保持原格式，或指定格式如"JPEG", "PNG"等

    返回:
    成功裁剪的图像路径列表
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理输入路径
    if isinstance(input_paths, str):
        # 如果是目录路径，获取目录下所有图像文件
        if os.path.isdir(input_paths):
            input_dir = input_paths
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
            input_paths = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and
                   any(f.lower().endswith(ext) for ext in extensions)
            ]
        else:
            # 如果是单个文件路径，转换为列表
            input_paths = [input_paths]

    # 确保输入路径列表不为空
    if not input_paths:
        raise ValueError("没有找到有效的图像文件")



    # 存储成功裁剪的图像路径
    cropped_paths = []

    for input_path in input_paths:
        # 处理目标尺寸
        target_width, target_height = target_size
        try:
            # 打开图像
            with Image.open(input_path) as img:
                # 确定输出格式
                output_format = format
                if format == "auto":
                    output_format = img.format if img.format else "JPEG"

                # 获取图像尺寸
                width, height = img.size

                # 计算裁剪区域
                if crop_mode == "center":
                    # 中心裁剪
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2
                elif crop_mode == "random":
                    # 随机位置裁剪
                    left = random.randint(0, max(0, width - target_width))
                    top = random.randint(0, max(0, height - target_height))
                elif crop_mode == "position" and position:
                    # 指定位置裁剪
                    left, top = position
                elif crop_mode == "top-left":
                    # 左上角裁剪
                    left = 0
                    top = 0
                elif crop_mode == "top-right":
                    # 右上角裁剪
                    left = max(0, width - target_width)
                    top = 0
                elif crop_mode == "bottom-left":
                    # 左下角裁剪
                    left = 0
                    top = max(0, height - target_height)
                elif crop_mode == "bottom-right":
                    # 右下角裁剪
                    left = max(0, width - target_width)
                    top = max(0, height - target_height)
                else:
                    # 默认中心裁剪
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2

                # 计算裁剪区域边界
                right = left + target_width
                bottom = top + target_height

                # 检查是否需要放大图像
                if not allow_upscaling and ((width < target_width or height < target_height) or (width < left and height < top)):
                    if scale is None:
                        print(
                            f"跳过 {input_path}: 原图尺寸 ({width}x{height}) 小于目标尺寸 ({target_width}x{target_height})")
                        continue
                    else:
                        # 比例裁剪
                        width_scale =  scale
                        height_scale = scale
                        target_width /=  width_scale
                        target_height /= height_scale
                        left /= width_scale
                        top /= height_scale
                        right = left + target_width
                        bottom = top + target_height
                        print(f'已裁剪: {input_path}: 原图尺寸 ({width}x{height}) 使用比例x{scale}裁剪')

                # 调整裁剪区域以确保在图像范围内
                if left < 0:
                    left = 0
                    right = min(target_width, width)

                if top < 0:
                    top = 0
                    bottom = min(target_height, height)

                if right > width:
                    right = width
                    left = max(0, width - target_width)

                if bottom > height:
                    bottom = height
                    top = max(0, height - target_height)

                # 执行裁剪
                cropped_img = img.crop((left, top, right, bottom))

                # 如果需要放大且原图小于目标尺寸
                if allow_upscaling and (cropped_img.width < target_width or cropped_img.height < target_height):
                    cropped_img = cropped_img.resize(target_size, Image.Resampling.LANCZOS)

                # 生成输出路径
                input_filename = os.path.basename(input_path)
                name, ext = os.path.splitext(input_filename)

                # 确定输出文件扩展名
                if format != "auto":
                    ext = f".{format.lower()}"
                elif not ext:
                    ext = ".jpg"  # 默认扩展名

                output_filename = f"{name}{suffix}{ext}"
                output_path = os.path.join(output_dir, output_filename)

                # 保存裁剪后的图像
                cropped_img.save(output_path, format=output_format)
                cropped_paths.append(output_path)

                print(f"已裁剪: {input_path} -> {output_path}")

        except Exception as e:
            print(f"处理 {input_path} 时出错: {str(e)}")

    return cropped_paths


import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Tuple, Optional


class ImageCropperGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("图像裁剪工具")
        self.root.geometry("900x700")

        # 变量初始化
        self.image_path = None
        self.image = None
        self.display_image = None
        self.original_image = None
        self.crop_rect = None
        self.dragging = False
        self.start_x, self.start_y = -1, -1
        self.end_x, self.end_y = -1, -1
        self.tk_image = None  # 存储Tkinter图像对象，防止被垃圾回收

        # 矩形框样式设置
        self.rect_color = (0, 255, 0)  # 绿色
        self.rect_thickness = 2

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 打开图像按钮
        self.open_btn = tk.Button(control_frame, text="打开GT图像", command=self.open_image, width=20)
        self.open_btn.pack(pady=5)

        # 选择输出目录按钮
        self.dir_btn = tk.Button(control_frame, text="选择输出目录", command=self.select_output_dir, width=20)
        self.dir_btn.pack(pady=5)

        # 矩形框颜色选择
        color_frame = tk.LabelFrame(control_frame, text="矩形框设置")
        color_frame.pack(fill=tk.X, pady=5)

        tk.Label(color_frame, text="颜色:").pack(anchor=tk.W)
        self.color_var = tk.StringVar(value="绿色")
        color_combo = ttk.Combobox(color_frame, textvariable=self.color_var,
                                   values=["绿色", "红色", "蓝色", "黄色", "白色"])
        color_combo.pack(fill=tk.X, pady=2)
        color_combo.bind("<<ComboboxSelected>>", self.change_rect_color)

        tk.Label(color_frame, text="线宽:").pack(anchor=tk.W)
        self.thickness_var = tk.IntVar(value=2)
        thickness_scale = tk.Scale(color_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                                   variable=self.thickness_var, command=self.change_rect_thickness)
        thickness_scale.pack(fill=tk.X, pady=2)

        # 保存选项
        save_frame = tk.LabelFrame(control_frame, text="保存选项")
        save_frame.pack(fill=tk.X, pady=5)

        self.save_gt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(save_frame, text="保存带矩形框的GT图", variable=self.save_gt_var).pack(anchor=tk.W)

        # 裁剪按钮
        self.crop_btn = tk.Button(control_frame, text="裁剪图像", command=self.crop_images,
                                  state=tk.DISABLED, width=20)
        self.crop_btn.pack(pady=5)

        # 重置按钮
        self.reset_btn = tk.Button(control_frame, text="重置选择", command=self.reset_selection, width=20)
        self.reset_btn.pack(pady=5)

        # 右侧图像显示区域
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 添加滚动条
        h_scrollbar = tk.Scrollbar(image_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        v_scrollbar = tk.Scrollbar(image_frame)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 图像显示画布
        self.canvas = tk.Canvas(image_frame, bg="gray",
                                xscrollcommand=h_scrollbar.set,
                                yscrollcommand=v_scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("请打开GT图像并拖动鼠标选择裁剪区域")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 输出目录
        self.output_dir = None

        # 设置颜色映射
        self.color_map = {
            "绿色": (0, 255, 0),
            "红色": (0, 0, 255),
            "蓝色": (255, 0, 0),
            "黄色": (0, 255, 255),
            "白色": (255, 255, 255)
        }

    def change_rect_color(self, event=None):
        color_name = self.color_var.get()
        self.rect_color = self.color_map.get(color_name, (0, 255, 0))
        if self.crop_rect:
            self.draw_rectangle()

    def change_rect_thickness(self, value):
        self.rect_thickness = int(value)
        if self.crop_rect:
            self.draw_rectangle()

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择GT图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp")]
        )

        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            if self.image is None:
                messagebox.showerror("错误", "无法打开图像文件")
                return

            self.original_image = self.image.copy()
            self.display_image = self.image.copy()

            # 更新画布尺寸
            self.update_canvas_size()
            self.update_display()

            self.status_var.set("图像已加载，请拖动鼠标选择裁剪区域")

    def update_canvas_size(self):
        """更新画布尺寸以适应图像大小"""
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.canvas.config(scrollregion=(0, 0, w, h))
            self.canvas.config(width=min(800, w), height=min(600, h))

    def select_output_dir(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_dir = dir_path
            self.status_var.set(f"输出目录: {dir_path}")

    def on_mouse_press(self, event):
        if self.image is None:
            return

        self.dragging = True
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.end_x, self.end_y = self.start_x, self.start_y

    def on_mouse_drag(self, event):
        if not self.dragging or self.image is None:
            return

        self.end_x, self.end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.draw_rectangle()

    def on_mouse_release(self, event):
        if not self.dragging or self.image is None:
            return

        self.dragging = False
        self.end_x, self.end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # 确保矩形是有效的
        x1, y1, x2, y2 = self.get_normalized_rect()
        if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # 最小尺寸检查
            self.crop_rect = (x1, y1, x2 - x1, y2 - y1)
            self.crop_btn.config(state=tk.NORMAL)
            self.status_var.set(
                f"已选择裁剪区域: {self.crop_rect[0]}, {self.crop_rect[1]}, {self.crop_rect[2]}, {self.crop_rect[3]}")
        else:
            self.crop_rect = None
            self.crop_btn.config(state=tk.DISABLED)
            self.status_var.set("选择区域太小，请重新选择")

    def get_normalized_rect(self):
        """确保矩形坐标是从左上到右下"""
        x1, x2 = min(self.start_x, self.end_x), max(self.start_x, self.end_x)
        y1, y2 = min(self.start_y, self.end_y), max(self.start_y, self.end_y)
        return x1, y1, x2, y2

    def draw_rectangle(self):
        if self.image is None:
            return

        # 复制原始图像
        self.display_image = self.original_image.copy()

        # 获取归一化矩形坐标
        x1, y1, x2, y2 = self.get_normalized_rect()

        # 在图像上绘制矩形
        cv2.rectangle(self.display_image, (int(x1), int(y1)), (int(x2), int(y2)),
                      self.rect_color, self.rect_thickness)

        # 更新显示
        self.update_display()

    def update_display(self):
        if self.display_image is None:
            return

        # 转换颜色空间 (BGR to RGB)
        display_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像
        pil_image = Image.fromarray(display_rgb)

        # 转换为Tkinter PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def reset_selection(self):
        if self.image is not None:
            self.display_image = self.original_image.copy()
            self.update_display()
            self.crop_rect = None
            self.crop_btn.config(state=tk.DISABLED)
            self.status_var.set("选择已重置，请重新选择裁剪区域")

    def save_gt_with_rectangle(self):
        """保存带有矩形框的GT图像"""
        if self.image_path is None or self.crop_rect is None:
            return None

        # 创建带矩形框的图像
        gt_with_rect = self.original_image.copy()
        x, y, w, h = self.crop_rect
        cv2.rectangle(gt_with_rect, (int(x), int(y)), (int(x + w), int(y + h)),
                      self.rect_color, self.rect_thickness)

        # 生成输出路径
        filename = os.path.basename(self.image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(self.output_dir, f"{name}{ext}")

        # 保存图像
        cv2.imwrite(output_path, gt_with_rect)
        return output_path

    def crop_images(self):
        if self.crop_rect is None or self.output_dir is None:
            messagebox.showwarning("警告", "请先选择裁剪区域和输出目录")
            return

        # 选择要裁剪的图像文件
        file_paths = filedialog.askopenfilenames(
            title="选择要裁剪的图像",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp")]
        )

        if not file_paths:
            return

        # 保存带矩形框的GT图像（如果选项被选中）
        gt_with_rect_path = None
        if self.save_gt_var.get():
            gt_with_rect_path = self.save_gt_with_rectangle()
            if gt_with_rect_path:
                self.status_var.set(f"已保存带矩形框的GT图像: {gt_with_rect_path}")

        # 获取裁剪参数
        x, y, w, h = self.crop_rect

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 裁剪图像
        success_count = 0
        for file_path in file_paths:
            try:
                # 打开图像
                img = cv2.imread(file_path)
                if img is None:
                    continue

                # 计算实际裁剪区域（考虑图像尺寸可能不同）
                img_h, img_w = img.shape[:2]
                scale_x = img_w / self.original_image.shape[1]
                scale_y = img_h / self.original_image.shape[0]

                # 计算在新图像上的裁剪区域
                new_x = int(x * scale_x)
                new_y = int(y * scale_y)
                new_w = int(w * scale_x)
                new_h = int(h * scale_y)

                # 确保裁剪区域在图像范围内
                new_x = max(0, min(new_x, img_w - 1))
                new_y = max(0, min(new_y, img_h - 1))
                new_w = min(new_w, img_w - new_x)
                new_h = min(new_h, img_h - new_y)

                # 执行裁剪
                cropped_img = img[new_y:new_y + new_h, new_x:new_x + new_w]

                # 生成输出路径
                filename = os.path.basename(file_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(self.output_dir, f"{name}_cropped{ext}")

                # 保存裁剪后的图像
                cv2.imwrite(output_path, cropped_img)
                success_count += 1

            except Exception as e:
                print(f"裁剪 {file_path} 时出错: {str(e)}")

        # 显示结果
        result_msg = f"成功裁剪 {success_count}/{len(file_paths)} 张图像"
        if gt_with_rect_path:
            result_msg += f"\n带矩形框的GT图像已保存: {os.path.basename(gt_with_rect_path)}"

        messagebox.showinfo("完成", result_msg)
        self.status_var.set(f"裁剪完成: {success_count}/{len(file_paths)} 张图像已保存到 {self.output_dir}")

    def run(self):
        self.root.mainloop()


# 独立裁剪函数
def crop_images(
        input_paths, output_dir, crop_rect,
        allow_upscaling=False, suffix="_cropped", format="auto"
):
    """
    裁剪一组图像到指定区域

    参数:
    input_paths: 输入图像路径列表
    output_dir: 输出目录
    crop_rect: 裁剪区域 (x, y, width, height)
    allow_upscaling: 是否允许放大图像
    suffix: 输出文件名后缀
    format: 输出格式
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理输入路径
    if isinstance(input_paths, str):
        if os.path.isdir(input_paths):
            input_dir = input_paths
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
            input_paths = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and
                   any(f.lower().endswith(ext) for ext in extensions)
            ]
        else:
            input_paths = [input_paths]

    if not input_paths:
        raise ValueError("没有找到有效的图像文件")

    x, y, w, h = crop_rect
    cropped_paths = []

    for input_path in input_paths:
        try:
            # 打开图像
            img = cv2.imread(input_path)
            if img is None:
                continue

            # 计算实际裁剪区域（考虑图像尺寸可能不同）
            img_h, img_w = img.shape[:2]
            scale_x = img_w / img.shape[1]  # 假设参考图像尺寸相同
            scale_y = img_h / img.shape[0]

            # 计算在新图像上的裁剪区域
            new_x = int(x * scale_x)
            new_y = int(y * scale_y)
            new_w = int(w * scale_x)
            new_h = int(h * scale_y)

            # 确保裁剪区域在图像范围内
            new_x = max(0, min(new_x, img_w - 1))
            new_y = max(0, min(new_y, img_h - 1))
            new_w = min(new_w, img_w - new_x)
            new_h = min(new_h, img_h - new_y)

            # 执行裁剪
            cropped_img = img[new_y:new_y + new_h, new_x:new_x + new_w]

            # 生成输出路径
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")

            # 保存裁剪后的图像
            cv2.imwrite(output_path, cropped_img)
            cropped_paths.append(output_path)

            print(f"已裁剪: {input_path} -> {output_path}")

        except Exception as e:
            print(f"处理 {input_path} 时出错: {str(e)}")

    return cropped_paths


def plot_training_trends(
        data_dict: Dict[str, List[float]],
        output_path: Optional[str] = None,
        title: str = "模型训练趋势",
        x_label: str = "迭代次数",
        y_label: str = "PSNR (dB)",
        y_range: Optional[List[float]] = None,
        x_range: Optional[List[float]] = None,
        dpi: int = 300,
        figsize: tuple = (10, 6)
):
    """
    绘制多个模型的训练趋势图

    参数:
    data_dict: 字典，键为模型名称，值为PSNR值列表
    iterations: 迭代次数列表
    output_path: 保存路径，如果为None则不保存
    title: 图表标题
    x_label: x轴标签
    y_label: y轴标签
    y_range: y轴范围，格式为[min, max]
    dpi: 图像分辨率
    figsize: 图像尺寸
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=figsize)

    # 设置白色网格背景样式
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # 定义标记样式和颜色
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))

    # 绘制每条曲线
    for i, (model_name, psnr_values) in enumerate(data_dict.items()):
        marker = markers[i % len(markers)]
        color = colors[i]
        if model_name=="RealESRGAN":
            color=[0.5,0.2,0.3,1]
            print(color)
        if model_name=="AMFDiff-2":
            color = [0.8, 0.8, 0.5, 1]
        if model_name=="AMFDiff-4":
            color = [0.7, 0.2, 0.8, 1]
        if model_name == 'ResShift':
            iterations = list(range(0, 140000, 10000)) if x_range is None else x_range
        else:
            iterations = list(range(0, 5000 * len(psnr_values), 5000)) if x_range is None else x_range
        ax.plot(
            iterations,
            psnr_values,
            marker=marker,
            markersize=6,
            linewidth=2,
            label=model_name,
            color=color,
            markevery=max(1, len(iterations) // 20)  # 避免标记过于密集
        )

    # 设置标题和轴标签
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # 设置坐标轴范围
    ax.set_xlim(min(iterations), max(iterations))
    if y_range:
        ax.set_ylim(y_range[0], y_range[1])

    # 添加图例
    ax.legend(loc='best', fontsize=10)

    # 优化布局
    plt.tight_layout()

    # 保存图像（如果指定了路径）
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path,"g.png"), dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"图像已保存至: {output_path}")

    # 显示图表
    plt.show()

    return fig, ax

def generate_lq_and_noize_img(gt_path):
    gt_img = util_image.imread(gt_path)
    gt_img_tensor = util_image.img2tensor(gt_img)
    lq_img_tensor = torch.nn.functional.interpolate(gt_img_tensor,scale_factor=0.3)
    noise = torch.randn_like(lq_img_tensor)
    lq_img_tensor_noise = lq_img_tensor + 0.13 * noise
    util_image.imwrite(util_image.tensor2img(lq_img_tensor),"temp/LQ.png",chn="bgr", dtype_in='float32')
    util_image.imwrite(util_image.tensor2img(lq_img_tensor_noise), "temp/LQ_noise.png",chn="bgr", dtype_in='float32')
# 使用示例
def visualize_attention_3d(attention_data,
                           decoder_layer_names=None,
                           resblock_names=None,
                           encoder_layer_names=None,
                           figsize=None,
                           cmap='hot',
                           vmin=0,
                           vmax=1,
                           annotate=True,
                           annotation_fontsize=8,
                           max_cols=3,
                           output_path="attention_picture",
                           dpi=300,
                           title="Attention Weights from Decoder to Encoder Layers"):
    """
    可视化三维注意力分布数据，自适应布局

    参数:
    attention_data: 3D numpy数组，形状为 [decoder_layers, resblocks, encoder_layers]
    decoder_layer_names: 解码器层名称列表，默认为数字索引
    resblock_names: resblock名称列表，默认为数字索引
    encoder_layer_names: 编码器层名称列表，默认为数字索引
    figsize: 图表大小
    cmap: 颜色映射
    vmin, vmax: 颜色范围
    annotate: 是否在热力图中添加数值注释
    annotation_fontsize: 注释字体大小
    annotation_color: 注释颜色
    title: 图表标题
    """

    # 获取各维度大小
    n_decoder_layers, n_resblocks, n_encoder_layers = attention_data.shape

    # 设置默认名称
    if decoder_layer_names is None:
        decoder_layer_names = [f"Decoder {i}" for i in range(n_decoder_layers)]
    if resblock_names is None:
        resblock_names = [f"Resblock {j+1}" for j in range(n_resblocks)]
    if encoder_layer_names is None:
        encoder_layer_names = [f"Encoder {k}" for k in range(n_encoder_layers)]

    # 计算子图布局 - 每行最多3个子图
    n_cols = min(max_cols, n_resblocks)
    n_rows = math.ceil(n_resblocks / n_cols)

    # 根据行数调整图形高度
    fig_height = max(4, 2 * n_rows)  # 每行至少2单位高度
    fig_width = min(18, 6 * n_cols)  # 每列最多6单位宽度

    # 创建row行col列的子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height) if figsize is None else figsize)
    fig.suptitle(title, fontsize=16)

    # 确保axes是二维数组，即使只有一行或一列
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 调整子图间距，为颜色条留出空间
    plt.subplots_adjust(right=0.85, wspace=0.2, hspace=0.2)

    # 为每个resblock创建热力图
    for resblock_idx in range(n_resblocks):
        # 计算子图位置
        row = resblock_idx // n_cols
        col = resblock_idx % n_cols

        # 提取当前resblock的数据
        resblock_data = attention_data[:, resblock_idx, :].T  # 转置以使编码器层在y轴

        # 绘制热力图
        im = axes[row, col].imshow(resblock_data, cmap=cmap, interpolation='nearest',
                                   vmin=vmin, vmax=vmax, aspect='auto')
        axes[row, col].set_title(f'{resblock_names[resblock_idx]}', fontsize=12)
        axes[row, col].set_xlabel('Decoder Layer', fontsize=10)
        axes[row, col].set_ylabel('Encoder Layer', fontsize=10)
        axes[row, col].set_xticks(range(n_decoder_layers))
        axes[row, col].set_xticklabels(decoder_layer_names)
        axes[row, col].set_yticks(range(n_encoder_layers))
        axes[row, col].set_yticklabels(encoder_layer_names)

        # 添加数值注释
        if annotate:
            for i in range(n_encoder_layers):
                for j in range(n_decoder_layers):
                    # 根据背景色选择文字颜色，确保可读性
                    bg_color = im.norm(resblock_data[i, j])
                    text_color = 'white' if bg_color > 0.5 else 'black'

                    text = axes[row, col].text(j, i, f'{resblock_data[i, j]:.3f}',
                                               ha="center", va="center",
                                               color=text_color,
                                               fontsize=annotation_fontsize)

    # 隐藏多余的子图
    for idx in range(n_resblocks, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # 添加颜色条 - 使用相对位置，确保不重叠
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight')
    # 保存图像（如果指定了路径）
    if output_path:
        parent = Path(output_path).parent
        parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"图像已保存至: {output_path}")
    plt.show()

    return fig, axes

def get_pictures(imgs_names_dir = "picture/my/sr"):
    '''
    根据 imgs_names_dir = "picture/my"中存放的图片，从其他模型图片文件夹里去搜集，分门别类放入到target_dir_name = "picture"

    Returns:

    '''
    input_dir = "result"
    target_dir_name = "picture"

    model_names = ["bicubic",
                   "bsrgan",
                   "esrgan",
                   "diffir",
                   "mybase_90000",
                   "realesr-gan",
                   "swinir",
                   "sinsr",
                   "diffbir",
                   "2layer-stu-16000-multi_step-k2",
                   "2layer_stu_30000-one_step"
                   # "all3-40000",
                   # "all3-2layer-55000",
                   # "mybase-2layer-90000"
                   ]
    exts = ['png', 'jpg', 'jpeg']
    possible_dirs = ["realset80", "realsr","imagenet-test","imagenet256","imagenet"]
    imgs_names = []
    gt_imgs_paths=["testdata/imagenet256/gt","testdata/RealSRx4/gt"]
    lq_imgs_paths=["testdata/imagenet256/lq","testdata/RealSRx4/lq","testdata/RealSet80"]
    for file_name in os.listdir(imgs_names_dir):
        imgs_names.append(file_name)

    for file_name in imgs_names:
        fn, ext = os.path.splitext(file_name)
        ext = ext.lstrip('.')
        target_dir = os.path.join(target_dir_name, fn)
        target_dir2 = os.path.join(target_dir, "sr")
        if not os.path.exists(target_dir2):
            os.makedirs(target_dir2)
        for model_name in model_names:
            found = False
            for dir_name in possible_dirs:
                if ext in exts:
                    extensions_to_try = [ext] + [e for e in exts if e != ext]
                else:
                    extensions_to_try = exts
                for new_ext in extensions_to_try:
                    try:
                        new_name = model_name + "." + new_ext
                        ori_name = fn + "." + new_ext
                        file_path = os.path.join(input_dir, model_name, dir_name, ori_name)
                        if os.path.exists(file_path):
                            target_path = os.path.join(target_dir2, new_name)
                            shutil.copy(file_path, target_path)
                            found = True
                            break  # 跳出扩展名循环
                    except Exception as e:
                        print(f"复制文件时出错: {e}")
                if found:
                    break  # 跳出目录循环
            if not found:
                print(f"未找到文件: {file_name} 对于模型 {model_name}")
        for gt_img_path in gt_imgs_paths:
            gt_file = os.path.join(gt_img_path,file_name)
            if os.path.exists(gt_file):
                new_name = "gt." + ext
                target_path = os.path.join(target_dir, new_name)
                shutil.copy(gt_file, target_path)
        for lq_imgs_path in lq_imgs_paths:
            lq_file = os.path.join(lq_imgs_path,file_name)
            if os.path.exists(lq_file):
                new_name = "lq." + ext
                target_path = os.path.join(target_dir, new_name)
                shutil.copy(lq_file, target_path)


def draw_rectangle(path, rectangle, out_path):
    # 读取图片
    image = cv2.imread(path)
    if image is None:
        print(f"错误：无法读取图片 {path}")
        return

    # 定义矩形框参数
    x1, y1, x2, y2 = rectangle
    # 确保坐标为整数
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    color = (0, 0, 255)  # BGR格式，红色
    thickness = 1  # 线宽

    # 绘制矩形框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    # 保存图片
    cv2.imwrite(out_path, image)


if __name__ == "__main__":

    # ================== 搜寻所有方法的图片 =============
    # get_pictures("picture/origin")

    # ==================裁剪 =========================
    scale = 4
    width = 200
    left = 250
    top = 650
    dir = "picture/realset80/Canon_004_LR4-yes"

    crop=True
    crop_images2(
        input_paths=dir+"/crop",
        output_dir=dir+"/crop2",
        target_size=(width ,width),
        position=(left,top),
        crop_mode="position",
        allow_upscaling=False,
        scale=4,
        suffix="_crop"
    )

    # # 画矩形框
    # r_left=left/scale
    # r_top=top/scale
    # r_width=width / scale
    # r_height=width / scale
    # rect = True
    # draw_rectangle(dir+"/crop/lq_crop.png",
    #                (r_left,r_top,r_left + r_width,r_top+r_height),
    #                out_path=dir+"/crop2/lq_crop.png")

    # with open(dir+"/crop_params.txt", 'a', encoding='utf-8') as file:
    #     content = ""
    #     if crop:
    #         content = f"============== crop ==============\ndir={dir}\nscale={scale}\nwidth={width}\nleft={r_left}\ntop={r_top}\n\n"
    #     if rect:
    #         content+= f"============== retangle ==============\n_left={r_left:.3f}\nr_top={r_top:.3f}\nr_width={r_width:.3f}\nr_height={r_height:.3f}\n"
    #     file.write(content)

    # =============== 裁剪工具启动GUI工具===================

    # app = ImageCropperGUI()
    # app.run()
    # get_pictures()
    # ================== 训练指标折线图 ===================
    # 绘制折线图

    # 生成模拟数据
    np.random.seed(42)
    data_dict = {
        "ESRGAN": [19.7058,20.0533,20.6407,20.8490,20.8057,20.7537,20.8417,21.1557,20.9710,20.8663,20.5279,20.5272,20.4914],
        "RealESRGAN":[21.9707,21.7990,21.6987,21.6658, 21.6789,21.6407,21.6552,21.5633,21.5941,21.5410,21.5304,21.4956,21.5237,21.5882],
        "DiffIR": [22.8640,22.8641,23.8982,23.9539,24.0792,24.1442,24.2124,24.2681,24.3295,24.3578,24.3895,24.4125,24.4376,24.4690,24.4628, 24.4667,24.4853,24.5086],
        "BSRGAN": [19.43,19.65,19.68,20.04,19.90,19.83, 20.22,19.79, 20.32, 20.31,20.49,20.24,19.98,20.43,19.98,19.69,20.44,19.98,19.56,20.18,19.89,20.18,19.96,19.92,20.00,20.26],
        "SwinIR": [22.74,22.89,23.03,23.09,23.11, 23.13,23.15, 23.18,23.20,23.19,23.22,23.19,23.19,23.14,23.18,23.16, 23.19,23.15, 23.12],
        "ResShift": [22.07,22.77,22.85,23.07,23.21,23.25,23.29,23.31,23.37,23.33,23.35,23.31,23.34,23.30],
        "AMFDiff-2":[21.6313,22.7831,23.0402,23.3052,23.3141,23.4666,23.4237,23.4668,23.5333,23.5624,23.5391,23.6403,23.6092,23.6122,23.6678,23.6710,23.6717,23.7431,23.6962,23.7145],
        "AMFDiff-4":[21.4744,22.7331,23.0820,23.2081,23.2775,23.3906,23.4561,23.4840, 23.5068,23.4388,23.5868,23.5535,23.5257,23.5591,23.5663,23.5171,23.5997, 23.5920, 23.5747,23.5300]
    }

    # 使用函数绘制图表
    # fig, ax = plot_training_trends(
    #     data_dict=data_dict,
    #     output_path="picture/g3",  # 保存为高分辨率PNG
    #     title="Validation PSNR curves of different models during training",
    #     x_label="iterations",
    #     y_label="PSNR(dB)",
    #     y_range=[19, 25],  # 设置Y轴范围
    #     dpi=1000  # 设置高分辨率
    # )
# ======================= 注意力图 ===============================
    data = np.zeros((4, 4, 4))

    # 注意力权重
    data[0, 0] = [0.1679, 0.0317, 0.0893, 0.7111]
    data[0, 1] = [0.7962, 0.0236, 0.0642, 0.116]
    data[0, 2] = [0.0001, 0.9992, 0.0002, 0.0006]
    data[1, 0] = [0.9996, 0.0001, 0.0001, 0.0002]
    data[1, 1] = [0.3709, 0.0988, 0.0243, 0.506]
    data[1, 2] = [0.9962, 0.0004, 0.0002, 0.0032]
    data[2, 0] = [0.0035, 0.0037, 0.9611, 0.0317]
    data[2, 1] = [0.9965, 0.0002, 0.0002, 0.0031]
    data[2, 2] = [0.0751, 0.7413, 0.1605, 0.0231]
    data[3, 0] = [0.0222, 0.6668, 0.2689, 0.0421]
    data[3, 1] = [0.4074, 0.3807, 0.1367, 0.0751]
    data[3, 2] = [0.4463, 0.1425, 0.2748, 0.1363]


    #  平均图
    # for i in range(data.shape[0]):
    #     avg[i, 0] = sum(data[i]) / data.shape[1]

    # 合图
    for i in range(4):
        data[i, 3] = sum(data[i]) / (data.shape[1] - 1)
    avg = np.zeros((4, 1, 4))

    # 自定义名称
    decoder_names = ["D1", "D2", "D3", "D4"]
    encoder_names = ["E1", "E2", "E3", "E4"]
    resblock_names = ["(a) Resblock-1", "(b) Resblock-2", "(c) Resblock-3","(d) Decoder-Encoder-Average"]


    # 调用函数
    # fig, axes = visualize_attention_3d(
    #     attention_data=data,
    #     decoder_layer_names=decoder_names,
    #     resblock_names=resblock_names,
    #     encoder_layer_names=encoder_names,
    #     figsize=(10, 10),
    #     cmap='YlOrRd',
    #     annotate=True,
    #     max_cols=2,
    #     annotation_fontsize=10,
    #     output_path="attention_picture/3/all-avg2.png",
    #     title="Layer-Attention Weights Distribution"
    # )
    # # 使用函数绘制图表
    # fig, ax = plot_training_trends(
    #     data_dict=data_dict,
    #     output_path="picture/g2",  # 保存为高分辨率PNG
    #     title="The attention distribution of the DFS ",
    #
    #     x_label="iterations",
    #     y_label="PSNR(dB)",
    #     y_range=[18, 25],  # 设置Y轴范围
    #     x_range=[],
    #     dpi=300  # 设置高分辨率
    # )


# ====================== 注意力随t变化的注意力图 =========================
    weights = np.array([
        # t = 3
        [
            [0.6867, 0.3077, 0.0051, 0.0005],  # Decoder 0
            [0.9441, 0.0479, 0.0074, 0.0006],  # Decoder 1
            [0.2482, 0.7044, 0.0444, 0.0029],  # Decoder 2
            [0.0999, 0.7977, 0.0589, 0.0434],  # Decoder 3
        ],
        # t = 2
        [
            [0.6558, 0.3319, 0.0103, 0.0020],
            [0.9380, 0.0533, 0.0078, 0.0009],
            [0.2177, 0.6952, 0.0682, 0.0189],
            [0.1034, 0.8229, 0.0557, 0.0179],
        ],
        # t = 1
        [
            [0.6847, 0.2941, 0.0173, 0.0039],
            [0.9263, 0.0635, 0.0087, 0.0015],
            [0.1841, 0.7167, 0.0779, 0.0214],
            [0.1156, 0.8076, 0.0548, 0.0220],
        ],
        # t = 0
        [
            [0.7645, 0.2243, 0.0082, 0.0030],
            [0.9641, 0.0234, 0.0084, 0.0042],
            [0.2434, 0.5912, 0.1615, 0.0039],
            [0.1542, 0.7153, 0.0825, 0.0480],
        ],
    ])
    # decoder_idx = 3  # 最底层 decoder
    # encoder_labels = ["Enc-1", "Enc-2", "Enc-3", "Enc-4"]
    # timesteps = [3, 2, 1, 0]
    #
    # plt.figure(figsize=(6, 4))
    # for enc_idx in range(4):
    #     plt.plot(
    #         timesteps,
    #         weights[:, decoder_idx, enc_idx],
    #         marker="o",
    #         label=encoder_labels[enc_idx]
    #     )
    #
    # plt.xlabel("Diffusion Step t")
    # plt.ylabel("Attention Weight")
    # plt.title("TDFS Temporal Evolution (Decoder 3)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
# 注意力权重分布 - 第一种
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
#     # 定义新的标签
#     decoder_labels = ['D1', 'D2', 'D3', 'D4']  # 解码器层：4（底层）到1（高层）
#     encoder_labels = ['E1', 'E2', 'E3', 'E4']  # 编码器层：1（浅层）到4（深层）
#     title = ["t=3","t=2","t=1","t=0"]
#     for i, ax in enumerate(axes):
#         sns.heatmap(
#             weights[i],
#             annot=True,
#             fmt=".2f",
#             cmap="viridis",
#             cbar=i == 3,
#             ax=ax,
#             xticklabels=encoder_labels,  # 设置编码器标签
#             yticklabels=decoder_labels,  # 设置解码器标签
#         )
#         # ax.set_title(f"t = {3 - i}")
#         ax.set_title(title[i])
#         ax.set_xlabel("Encoder")
#         ax.set_ylabel("Decoder")
#         ax.set_xticks(np.arange(4) + 0.5)
#         ax.set_yticks(np.arange(4) + 0.5)
#     plt.tight_layout()
#     output_path = "p/tdfs_attention_heatmaps4.png"
#     if output_path:
#         parent = Path(output_path).parent
#         parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(output_path, dpi=1000, bbox_inches='tight', facecolor='white')
#     plt.show()


    # 第二种 单个折线图
    # decoder_idx = 3  # 最底层 decoder
    # encoder_labels = ["Enc-1", "Enc-2", "Enc-3", "Enc-4"]
    # timesteps = [3, 2, 1, 0]
    #
    # plt.figure(figsize=(6, 4))
    # for enc_idx in range(4):
    #     plt.plot(
    #         timesteps,
    #         weights[:, decoder_idx, enc_idx],
    #         marker="o",
    #         label=encoder_labels[enc_idx]
    #     )
    #
    # plt.xlabel("Diffusion Step t")
    # plt.ylabel("Attention Weight")
    # plt.title("TDFS Temporal Evolution (Decoder 3)")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()