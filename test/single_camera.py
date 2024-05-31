import cv2
from ultralytics import YOLO
import time


def main():
    # Load the YOLOv8 model
    model = YOLO(r'D:\workingSpace\Projects\train240502\runs\detect\train\weights\best.pt')
    # 打开四个相机

    cap = cv2.VideoCapture(0)
    # 设置相机分辨率为1920x1080

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 检查相机是否成功打开

    if not cap.isOpened():
        print(f"无法打开相机 ")
        return

    # 初始化时间
    prev_time = time.time()
    while True:
        # 计算实时 FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time

        # 避免除以零错误
        if elapsed_time != 0:
            fps = 1 / elapsed_time
        else:
            fps = 0

        # 逐个相机捕获视频并显示

        ret, frame = cap.read()
        if not ret:
            print(f"无法读取相机 {i} 的视频帧")
            continue

        # 缩小图像以适应屏幕大小
        resized_frame = cv2.resize(frame, (480, 270))
        results = model(resized_frame)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # 在帧上绘制实时 FPS
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示缩小后的图像
        cv2.imshow(f'Camera', annotated_frame)

        # 检查是否按下了'q'键，如果是则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机对象
    cap.release()

    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
