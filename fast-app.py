import os
import uuid
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import gradio as gr

import app as ap

# 从你的推理模块导入（确保路径正确）
from inference import run_inference

# 创建 FastAPI 实例
fastapi_app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = "api/uploads"
OUTPUT_DIR = "api/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 定义 FastAPI 路由 =================
@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

@fastapi_app.post("/super_resolution")
async def super_resolution(file: UploadFile = File(...), task: str = "realsr", scale: int = 4):
    try:
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.png")
        with open(input_path, "wb") as f:
            f.write(await file.read())
        output_dir = os.path.join(OUTPUT_DIR, file_id)
        output_path = run_inference(
            input_path=input_path,
            output_dir=output_dir,
            task=task,
            scale=scale,
            version="v3"
        )
        return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        return {"error": str(e)}

@fastapi_app.get("/", response_class=HTMLResponse)
async def default_page():
    # 注意：Gradio 现在挂载在 /gradio 路径下，使用相对路径即可
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>超分服务</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
                h1 { color: #2c3e50; }
                p { color: #34495e; }
                .button { display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px; }
                .button:hover { background-color: #2980b9; }
            </style>
        </head>
        <body>
            <h1>AMFDiff 超分服务</h1>
            <p>FastAPI 后端已启动，访问 <a href="/docs">/docs</a> 查看 API 文档。</p>
            <p>Gradio 交互界面已挂载：</p>
            <a href="/gradio" class="button" target="_blank">打开 Gradio 界面</a>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ================= 挂载 Gradio 应用 =================
# 将 gradio_demo 挂载到 /gradio 路径下，并得到最终的 app 对象
final_app = gr.mount_gradio_app(fastapi_app, ap.demo, path="/gradio")

# ================= 启动服务 =================
if __name__ == "__main__":
    uvicorn.run(final_app, host="127.0.0.1", port=8000)