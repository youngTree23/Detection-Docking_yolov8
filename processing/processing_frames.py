import time
from collections import deque

import cv2
import numpy as np

from camera import FOCAL_LENGTH_MM, SENSOR_WIDTH_MM


class FrameProcessor:
    def __init__(self, model, img_shape, buffer_size=5, azimuth_threshold=20):
        self.model = model
        self.frame = None
        self.previous_time = time.time()
        self.img_width, self.img_height = img_shape
        self.azimuth_buffer = deque(maxlen=buffer_size)
        self.azimuth_threshold = azimuth_threshold
        self.clear_count = 0

    @property
    def current_frame(self):
        return self.frame

    def clear_azimuth_buffer(self):
        self.clear_count += 1

        if self.clear_count % 10 == 0:
            self.azimuth_buffer.clear()
        if self.clear_count == 50:
            self.clear_count = 0

    def capture_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            print("无法读取相机的视频帧")
            self.frame = None
        else:
            self.frame = frame

    def calculate_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.previous_time
        fps = 1 / elapsed_time if elapsed_time else float('inf')
        self.previous_time = current_time
        return fps

    def overlay_fps(self, frame):
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def process_frame(self, camera_idx, camera_angle, azimuths: list):
        if self.frame is None:
            return np.zeros((self.img_height, self.img_width, 3), np.uint8), None
        # 初始化 annotated_frame
        results = self.model.predict(self.frame, conf=0.7, device='cpu', max_det=1)
        result = results[0]
        xywh = None
        annotated_frame = result.plot()
        if len(result.boxes.xywh) > 0:
            conf = result.boxes.conf.item()
            xywh = result.boxes.xywh.squeeze().tolist()
            x_center, y_center = xywh[0], xywh[1]
            w_pixel = xywh[2]
            cv2.circle(annotated_frame, (int(x_center), int(y_center)), 10, (255, 0, 0), -1)
            angle = self.calculate_azimuth(x_center, camera_angle)
            if self.is_valid_azimuth(angle):
                azimuths.append(angle)
            distance = self.estimate_distance(w_pixel)
            print("********************")
            print(f"{camera_idx + 1}号相机检测到目标，\n置信度为{np.round(conf, 2)},\n大致距离为{distance}m")
        self.overlay_fps(annotated_frame)
        return annotated_frame

    def process_multiple_cameras(self, caps, camera_angles):
        annotated_frames = []
        azimuths = []
        for i, cap in enumerate(caps):
            self.capture_frame(cap)
            if self.current_frame is not None:
                annotated_frame = self.process_frame(i, camera_angles[i], azimuths)
                annotated_frames.append(annotated_frame)
            else:
                annotated_frames.append(np.zeros((self.img_height, self.img_width, 3), np.uint8))

        if azimuths:
            azimuth = self.compute_average_azimuth(azimuths)  # 对所计算方位角进行合法性检验，并求均值
            self.azimuth_buffer.append(azimuth)  # 加入缓存区，挤掉最老数据
            return annotated_frames, azimuth
        return annotated_frames, None

    def calculate_azimuth(self, x_center, camera_angle):
        img_width = self.img_width
        theta = ((x_center - (img_width / 2)) / img_width * 135) + camera_angle
        return theta

    def is_valid_azimuth(self, new_azimuth):
        if not self.azimuth_buffer:
            return True
        for azimuth in self.azimuth_buffer:
            if abs(azimuth - new_azimuth) < self.azimuth_threshold:
                return True
        return False

    def estimate_distance(self, w_pixels, w_actual=1.6):
        """
        估算目标物体距离的函数

        参数:
        - W_actual: 目标物体的实际宽度，以米为单位
        - W_pixels: bounding box的水平宽度，以像素为单位

        返回:
        - 估算的目标物体距离，以米为单位
        """
        # 计算相机的焦距（以像素为单位）
        focal_length_pixels = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * self.img_width

        # 计算目标物体的距离
        distance = np.round((w_actual * focal_length_pixels) / w_pixels, 2)

        return distance

    @staticmethod
    def compute_average_azimuth(azimuths):
        """
        处理初步计算所得方位角，将角度差值小于10度的视为合法数据，并求平均值。
        :param azimuths: list of float
        :return: float
        """
        if len(azimuths) > 1:
            valid_azimuths = []
            for i in range(len(azimuths) - 1):
                for j in range(i + 1, len(azimuths)):
                    if abs(azimuths[i] - azimuths[j]) < 10:
                        valid_azimuths.append((azimuths[i] + azimuths[j]) / 2)
            if valid_azimuths:
                return np.mean(valid_azimuths)
        return azimuths[0]

    @staticmethod
    def display_single_camera(frame, index):
        cv2.imshow(f'Camera {index}', frame)

    @staticmethod
    def display_multiple_cameras(frames, shape, window_name="Cameras", grid_size=(2, 2)):
        rows, cols = grid_size
        height, width = shape

        grid_image = np.zeros((height * rows, width * cols, 3), np.uint8)

        for i, frame in enumerate(frames):
            resized_frame = cv2.resize(frame, (width, height))
            row = i // cols
            col = i % cols
            grid_image[row * height:(row + 1) * height, col * width:(col + 1) * width] = resized_frame

        cv2.imshow(window_name, grid_image)
