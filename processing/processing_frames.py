import cv2
import time


class FrameProcessor:
    def __init__(self, model):
        self.model = model
        self.frame = None
        self.previous_time = time.time()

    def get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机的视频帧")
        self.frame = frame

    def get_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.previous_time
        # 避免除以零错误
        if elapsed_time != 0:
            fps = 1 / elapsed_time
        else:
            fps = 0

        self.previous_time = current_time
        return fps

    def display_fps(self, frame):
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def process_frame(self, camera_name, camera_angle):
        img_width = 480
        img_height = 270
        # 缩小图像以适应屏幕大小
        resized_frame = cv2.resize(self.frame, (img_width, img_height))
        # 使用模型检测物体
        results = self.model(resized_frame)
        # 可视化检测结果
        annotated_frame = results[0].plot()
        self.display_fps(annotated_frame)
        # 显示缩小后的图像
        cv2.imshow(f'Camera {camera_name}', annotated_frame)

        # 如果检测到物体，则打印检测信息
        if len(results[0].boxes.xyxy) != 0:
            # 计算Bounding box中心点的方位角
            for result in results:
                conf = result.probs
                print(f"置信度{conf}")
                for xyxy in result.boxes.xyxy.tolist():
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    # 根据相机的角度调整方位角
                    theta = ((x_center - img_width / 2) / img_width * 135) + camera_angle
                    print(f"{camera_name}号相机检测到目标, 水平方位角为{theta}")
        else:
            print(f"{camera_name}号相机：原地不动")
