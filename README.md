说明
=====
本仓库包含一个用于 AimLab 自动瞄准演示的脚本 aimbot.py。该 README 说明如何在 Windows 环境下安装依赖并运行脚本，以及脚本的快捷键说明和建议的依赖版本。

关于文件
----
train.py是用来训练模型的，将data=r'D:\pycharm\aimlab-ainbot\AimLab_ball.v1i.yolov8\data.yaml'改为你的地址。
aimbot.py是自动瞄准脚本。

注意
----
- 脚本目前仅在 Windows 上测试（使用了 Win32 鼠标事件）。
- 若使用 GPU，请确保已正确安装对应版本的 CUDA 驱动和与之匹配的 PyTorch（下文给出示例安装命令）。
- 运行脚本会操控鼠标，请在安全环境中测试。

推荐环境
--------
- 操作系统：Windows 10/11
- Python：3.8 - 3.10（建议使用 conda 管理环境）

建议的依赖版本（已知可用/兼容）
----------------------------
- torch         == 2.0.1+cu118 （或根据你的 CUDA 版本选择合适的 torch build）
- torchvision   == 0.15.2+cu118
- ultralytics   >= 8.0.0
- opencv-python == 4.7.0
- dxcam         == 0.0.8
- pynput        == 1.7.6
- tkinter       (随 Python 自带，无需 pip 安装)


安装步骤（conda 示例）
--------------------
1) 创建并激活 conda 环境（可改为 virtualenv）

   conda create -n yolo python=3.9 -y
   conda activate yolo

2) 安装 PyTorch（含 CUDA，以下示例为 CUDA 11.8）：

   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html

   如果没有 GPU 或不想安装 CUDA，请使用：
   pip install torch torchvision

3) 安装其余依赖：

   pip install ultralytics opencv-python dxcam pynput

4) 确认权重文件存在：

   - 默认脚本会加载项目目录下的 yolov8s.pt（可替换为 yolov8n.pt 等）。
   - 请确保权重文件存在于项目目录，或在脚本中修改为正确路径。

运行脚本
-------
1) 打开命令行，激活 conda 环境：
   conda activate yolo

2) 在项目根目录运行：
   python aimbot.py

3) 快捷键：
   - 按 A：启用自动瞄准（ON）
   - 按 Q：停止自动瞄准（OFF）
   - 按 ESC：退出脚本并停止所有线程

主要配置项（在 aimbot.py 顶部可调整）
----------------------------------
- CONFIDENCE_THRESHOLD：检测置信度阈值，调低可提高召回但可能增加误报
- MIN_BOX_SIZE / MAX_BOX_SIZE：过滤过小或过大的检测框

推理/速度相关（当前脚本已实现推理节流与结果复用）：
- INFER_IMGSZ：推理输入尺寸（越大越准但越慢）
- INFER_MAX_DET：最大检测数量（越大越慢）
- INFER_MAX_FPS：推理最大帧率限制，用于防止 GPU/CPU 被推理占满
- INFER_EVERY_N_FRAMES：每 N 帧推理一次，N 越大越快但更新更“跳”

瞄准/射击相关：
- SMOOTH_FACTOR：鼠标移动平滑系数（越大越快，过大可能抖动）
- DEADZONE_RADIUS：死区（像素），小于该距离不再移动以减少抖动
- AIM_OFFSET_X / AIM_OFFSET_Y：瞄点偏移校准

FP16（float16）说明
-----------------
- 脚本在检测到 GPU 时，会在 model.predict 中设置 half=True 以开启 FP16 推理。
- 若 GPU/驱动不支持 FP16 或出现异常，可将 half 相关逻辑关闭（改为 half=False）。



常见问题
--------
- FileNotFoundError: 找不到权重文件（yolov8s.pt 等）——请把权重放到项目目录或修改脚本中的路径。
- GPU 未使用：确认 CUDA 驱动、NVIDIA 驱动已安装，并使用与 CUDA 版本匹配的 torch 安装包。

安全与合规
----------
- 本脚本会自动移动并点击鼠标，请仅在允许和安全的环境中测试。
- 请勿在在线游戏中使用此类脚本，以免违反游戏规则或法律法规。

