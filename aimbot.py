
import dxcam
import cv2
import queue
import threading
import ctypes
import time
from ultralytics import YOLO
from pynput import keyboard
import tkinter as tk
import torch

# 常量
AIMLAB_WINDOW_NAME = "aimlab_tb"
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
CLICK_DISTANCE_THRESHOLD = 400  # 距离阈值
CONFIDENCE_THRESHOLD = 0.3  # 置信度
MIN_BOX_SIZE = 15  # 最小框尺寸
MAX_BOX_SIZE = 400  # 限制最大框尺寸，防止画一个屏幕那么大的框

# DPI缩放
DPI_ZOOM = 1.0

# 平滑移动设置 (1.0 = 瞬移)
SMOOTH_FACTOR = 0.5  # 拉枪速度
AIM_SENSE = 1.0  # 瞄准灵敏度系数
DEADZONE_RADIUS = 3  # 死区像素

# 瞄点校准
AIM_OFFSET_X = -3  # 像素
AIM_OFFSET_Y = -2  # 像素


# 全局变量
running = True
enabled = False

# 用于屏幕叠加层传递数据的队列
overlay_queue = queue.Queue(maxsize=1)

def on_press(key):
    global running, enabled
    if key == keyboard.Key.esc:
        print("检测到 ESC")
        running = False
        return False

    try:
        ch = getattr(key, "char", None)
        if ch:
            ch = ch.lower()
            if ch == "a":
                enabled = True
                print("ON")
            elif ch == "q":
                enabled = False
                print("OFF")
    except Exception:
        pass

# 加载YOLO模型
model = YOLO(model="yolov8s.pt")

# 若有 GPU，启用 cudnn.benchmark 并把模型移动到 GPU，尝试半精度加速
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model.to('cuda')
        print('使用 GPU')
    else:
        print('未检测到 GPU，使用 CPU')
except Exception as e:
    print(f'模型迁移到 GPU 失败: {e}')

frame_queue = queue.Queue(maxsize=1)

def GetScreenRegion():
    width = ctypes.windll.user32.GetSystemMetrics(0)
    height = ctypes.windll.user32.GetSystemMetrics(1)
    left, top = 0, 0
    right, bottom = width, height
    region = (left, top, right, bottom)
    window_center_x = width // 2
    window_center_y = height // 2
    return region, (left, top, width, height), (window_center_x, window_center_y)

def CaptureScreen_thread_func():
    region, rect, window_center = GetScreenRegion()
    camera = dxcam.create()  # 默认捕捉全屏幕
    camera.start(target_fps=144) # 提高采集率从30提升到144
    while running:
        frame = camera.get_latest_frame()
        if frame is None:
            continue
        # 转换为BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put((frame_bgr, rect, window_center))
    camera.stop()

def Overlay_thread_func():
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-transparentcolor", "white")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.config(bg="white")

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="white", highlightthickness=0)
    canvas.pack()

    # 必须先 update() 才能获取到准确的 window id 来设置鼠标穿透
    root.update()

    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    if not hwnd:
        hwnd = root.winfo_id()
    GWL_EXSTYLE = -20
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_LAYERED = 0x00080000
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_TRANSPARENT | WS_EX_LAYERED)

    # 缓存上一次的绘制数据，当没有新画面時保持显示 ON/OFF 状态
    last_boxes = []

    def update_overlay():
        nonlocal last_boxes
        if not running:
            root.destroy()
            return

        try:
            boxes_data, target_data, target_size, en_state = overlay_queue.get_nowait()
            last_boxes = boxes_data
        except queue.Empty:
            # 如果没有新帧，仅获取当前的 enabled 状态持续刷新
            en_state = enabled

        canvas.delete("all")

        for (gx, gy, gw, gh) in last_boxes:
            canvas.create_rectangle(gx, gy, gx + gw, gy + gh, outline="#00ff00", width=2)

        # 状态文字
        canvas.create_text(20, 30, text=f"ENABLED: {en_state}  (A=on, Q=off, ESC=quit)", fill="yellow", font=("Arial", 12, "bold"), anchor="w")

        root.after(10, update_overlay)

    root.after(10, update_overlay)
    root.mainloop()

def YoloPredict_thread_func():
    locked_target = None  # 添加目标锁定状态，记录上一次锁定的目标中心
    TRACKING_RADIUS_SQ = 100 * 100  # 追踪检测半径（平方面积），适度缩小以避免把旁边的小球当成原目标

    # 引入小数像素残差累计，解决整数截断导致的微小抖动
    residual_x = 0.0
    residual_y = 0.0

    last_click_time = 0.0  # 记录上次开枪时间，防止对同一个正在消失的目标连发导致命中率下降

    while running:
        try:
            frame_data = frame_queue.get(timeout=0.1)
            frame, rect, window_center = frame_data
            left, top, width, height = rect
            window_center_x, window_center_y = window_center

            # YOLO推理（GPU 下用 half=True 开启 FP16 推理；不要手动 .half()，避免 dtype 混用）
            use_half = bool(torch.cuda.is_available())
            with torch.inference_mode():
                results = model.predict(
                    source=frame,
                    save=False,
                    show=False,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=1280,
                    max_det=200,
                    half=use_half,
                    verbose=False
                )
            detections = results[0].boxes
            overlay_boxes = []
            final_target = None
            target_size = None

            if len(detections) > 0:
                current_valid_targets = []

                # 遍历收集所有有效框
                for box in detections:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x, y, w, h = int(x1.item()), int(y1.item()), int(x2.item() - x1.item()), int(y2.item() - y1.item())

                    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE or w > MAX_BOX_SIZE or h > MAX_BOX_SIZE:
                        continue

                    conf = float(box.conf[0].item())

                    center_x = x + w // 2
                    center_y = y + h // 2

                    g_x, g_y = left + x, top + y
                    overlay_boxes.append((g_x, g_y, w, h))
                    current_valid_targets.append((center_x, center_y, w, h, conf))

                if not current_valid_targets:
                    locked_target = None  # 视野内无有效目标，释放锁定
                else:
                    best_target = None

                    # 1. 优先尝试维持现有的锁定目标 (结合置信度)
                    if locked_target is not None:
                        pred_x, pred_y = locked_target[0], locked_target[1]

                        best_dist = float('inf')
                        for tgt in current_valid_targets:
                            cx, cy, tw, th, conf = tgt
                            # 计算与预测位置的距离
                            dist_to_lock = (cx - pred_x)**2 + (cy - pred_y)**2
                            if dist_to_lock < TRACKING_RADIUS_SQ:
                                # 纯依靠距离锁定，不再混配置信度计算防止抢夺准星
                                if dist_to_lock < best_dist:
                                    best_dist = dist_to_lock
                                    best_target = tgt

                    # 2. 如果之前没有锁定目标，或者上一次锁定的目标已丢失，则重新寻找距离屏幕中心最近且置信度高的
                    if best_target is None:
                        best_dist = float('inf')
                        for tgt in current_valid_targets:
                            cx, cy, tw, th, conf = tgt
                            dist_to_center = (cx - window_center_x)**2 + (cy - window_center_y)**2
                            # 纯依靠距离中心远近寻找新目标
                            if dist_to_center < best_dist:
                                best_dist = dist_to_center
                                best_target = tgt

                    # 3. 处理最终选定的目标
                    if best_target is not None:
                        final_x_raw, final_y_raw, w, h, conf = best_target

                        final_x = int(final_x_raw)
                        final_y = int(final_y_raw)

                        # 仅应用固定偏移校准
                        final_x += AIM_OFFSET_X
                        final_y += AIM_OFFSET_Y

                        # （已回退）不再做左右对称拉回
                        # dx_to_center = final_x - window_center_x
                        # final_x = int(round(final_x - dx_to_center * AIM_SYMMETRIC_PULL_X))

                        # 防止偏移后超出画面
                        final_x = max(0, min(width - 1, final_x))
                        final_y = max(0, min(height - 1, final_y))

                        locked_target = (final_x, final_y)  # 更新锁定状态

                        dx_best = final_x - window_center_x
                        dy_best = final_y - window_center_y
                        distance_sq = dx_best**2 + dy_best**2
                        min_distance = distance_sq  # 兼容后文点击距离判断

                        global_x = left + final_x
                        global_y = top + final_y
                        final_target = (global_x, global_y)
                        target_size = (w, h)

                        # 在外层先计算好开火半径
                        click_radius = min(w, h) * 0.25
                        click_radius = max(3.0, click_radius)

                        # 开启状态下才执行瞄准和点击
                        if enabled:
                            # 增加死区判断：如果距离目标中心过远，进行平滑拉枪
                            if distance_sq > DEADZONE_RADIUS * DEADZONE_RADIUS:
                                move_x_float = dx_best * SMOOTH_FACTOR * DPI_ZOOM * AIM_SENSE + residual_x
                                move_y_float = dy_best * SMOOTH_FACTOR * DPI_ZOOM * AIM_SENSE + residual_y

                                move_x = int(move_x_float)
                                move_y = int(move_y_float)

                                # 记录剩余的小数部分留到下一帧
                                residual_x = move_x_float - move_x
                                residual_y = move_y_float - move_y

                                if move_x != 0 or move_y != 0:
                                    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, move_x, move_y, 0, 0)
                            else:
                                residual_x = 0.0
                                residual_y = 0.0

                            current_time = time.time()
                            # 判定开火，附带冷却时间避免对旧球多次射击
                            if distance_sq < click_radius ** 2 and (current_time - last_click_time) > 0.12:
                                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                                time.sleep(0.015)
                                ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                                time.sleep(0.01)
                                last_click_time = time.time()

                            # 重要：不管有没有点击，只要发生了视角移动或射击，必须清空旧的帧队列保证下一捕捉是最新的！
                            with frame_queue.mutex:
                                frame_queue.queue.clear()

                    pass
            overlay_queue.put((overlay_boxes, final_target, target_size, enabled))

        except queue.Empty:
            continue
        except Exception as e:
            print(f"鼠标操作或推理失败: {e}")
            continue

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    t1 = threading.Thread(target=CaptureScreen_thread_func, daemon=True)
    t2 = threading.Thread(target=YoloPredict_thread_func, daemon=True)
    t1.start()
    t2.start()

    try:
        # Tkinter必须在主线程中运行，否则界面无法正常显示或失去响应
        Overlay_thread_func()
    except Exception as e:
        print(f"Overlay线程启动失败: {e}")

    try:
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        listener.stop()
        print("程序结束")
