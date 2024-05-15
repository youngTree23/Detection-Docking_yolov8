import cv2
from ultralytics import YOLO
import time
import numpy as np


# 获取每个相机的帧
def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("无法读取相机的视频帧")
        return None
    return frame


# 处理每个相机的帧
def process_frame(frame, model, camera_name, camera_parameters, prev_time):
    # 缩小图像以适应屏幕大小
    resized_frame = cv2.resize(frame, (480, 270))
    # 使用模型检测物体
    results = model(resized_frame)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    # 获取当前时间并计算帧速率
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time

    # 避免除以零错误
    if elapsed_time != 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    # 在帧上绘制实时 FPS
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 显示缩小后的图像
    cv2.imshow(f'Camera {camera_name}', annotated_frame)
    # 如果检测到物体，则计算方位角
    if len(results[0].boxes.xyxy) != 0:
        # 计算Bounding box中心点的方位角
        for xyxy in results[0].boxes.xyxy.tolist():
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            # 将中心点坐标从图像坐标系转换到相机坐标系
            x_camera = (x_center - camera_parameters['cx']) / camera_parameters['fx']
            y_camera = (y_center - camera_parameters['cy']) / camera_parameters['fy']
            # 计算方位角
            azimuth_radians = np.arctan2(x_camera, camera_parameters['focal_length_x'])
            azimuth_degrees = np.degrees(azimuth_radians)
            print(f"{camera_name}号相机检测到目标，方位角为{azimuth_degrees}度")

    return prev_time


# 主函数
def main():
    # Load the YOLOv8 model
    model = YOLO(r'D:\workingSpace\Projects\train240502\runs\detect\train\weights\best.pt')
    # 打开四个相机
    num_cameras = 4
    caps = [cv2.VideoCapture(i) for i in range(1, num_cameras + 1)]

    # 设置相机分辨率为640x480
    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 检查相机是否成功打开
    for i, cap in enumerate(caps, start=1):
        if not cap.isOpened():
            print(f"无法打开相机 {i}")
            return

    # 定义相机内参和安装姿态
    camera_parameters = {
        'fx': 1000,  # x轴焦距
        'fy': 1000,  # y轴焦距
        'cx': 320,  # 光心x坐标
        'cy': 240,  # 光心y坐标
        'focal_length_x': 1000  # x轴焦距（假设与x轴焦距相同）
    }

    # 初始化时间
    prev_time = time.time()

    # 顺序处理每个相机
    while True:
        for i, cap in enumerate(caps, start=1):
            frame = get_frame(cap)
            if frame is None:
                continue
            prev_time = process_frame(frame, model, i, camera_parameters, prev_time)

        # 等待按键输入，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机对象
    for cap in caps:
        cap.release()

    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
