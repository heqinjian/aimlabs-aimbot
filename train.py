from ultralytics import YOLO
import torch


def train_model():
    # 1. 环境检查
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"[*] 当前使用设备: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    # 2. 加载模型基座
    model = YOLO('yolov8s.pt')

    # 3. 开始训练
    model.train(
        data=r'D:\pycharm\aimlab-ainbot\AimLab_ball.v1i.yolov8\data.yaml',
        epochs=150,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,  # Windows下如果报错可以设为0

        # --- 核心参数修正 ---
        patience=0,  # 禁用早停，强制练满
        save=True,
        pretrained=True,
        optimizer='SGD',
        lr0=0.01,
        cos_lr=True,

        # --- 数据增强参数  ---
        mosaic=1.0,  # 必须开启，针对小目标
        mixup=0.1,
        degrees=5.0,
        shear=2.0,  # 剪切变换
        perspective=0.0001,  # 透视变换
        flipud=0.0,  # 上下翻转
        fliplr=0.5  # 左右翻转
    )

    print("[✔] 训练完成！")


if __name__ == '__main__':
    train_model()