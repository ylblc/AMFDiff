import os
import uuid

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import torch
from inference import *

from api.utils.inference import run_inference

app = FastAPI()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


# 推理接口
@app.post("/super_resolution")
async def super_resolution(file: UploadFile = File(...),task:str="realsr",scale:int=4):
    try:
        # ===== 1. 保存上传图片 =====
        file_id = str(uuid.uuid4())
        input_path = os.path.join(UPLOAD_DIR, f"{file_id}.png")

        with open(input_path, "wb") as f:
            f.write(await file.read())

        # ===== 2. 推理 =====
        output_dir = os.path.join(OUTPUT_DIR, file_id)

        output_path = run_inference(
            input_path=input_path,
            output_dir=output_dir,
            task=task,
            scale=scale,
            version="v3"
        )

        # ===== 3. 返回结果 =====
        return FileResponse(output_path, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}


# ===== 5. 启动 =====
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)