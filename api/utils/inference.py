from inference_resshift import *



def run_inference(
    input_path,
    output_dir,
    task="realsr",
    scale=4,
    version="v3"
):
    """
    封装原有推理逻辑，供外部调用
    """
    os.makedirs(output_dir, exist_ok=True)

    # ===== 模拟命令行参数 =====
    sys.argv = [
        "inference_resshift.py",
        "-i", input_path,
        "-o", output_dir,
        "--task", task,
        "--scale", str(scale),
        "--version", version
    ]

    # ===== 调用原脚本 =====
    main(sys.argv)
    # ===== 默认输出路径（按你项目实际改）=====
    output_path = os.path.join(output_dir, "result.png")

    return output_path