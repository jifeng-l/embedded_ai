import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
import time

# Florence-2 设置
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# 视频输入：摄像头（0）或本地文件路径
video_path = 0  # 或 "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] 无法打开视频源")
    exit()

print("[INFO] 开始视频检测... 按 q 退出")

# 控制处理帧率（每 N 帧处理一次）
FRAME_SKIP = 5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # 跳帧加速

    # 转为 PIL 图像
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 模型推理
    inputs = processor(text="<OD>", images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

    # 绘制结果
    for bbox, label in zip(result['<OD>']['bboxes'], result['<OD>']['labels']):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

    # 显示
    cv2.imshow('Florence-2 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()