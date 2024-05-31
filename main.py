import cv2
import numpy as np
from ultralytics import YOLO

from camera import CameraManager, CAMERA_ANGLES
from processing import FrameProcessor


def main():
    # 加载YOLOv8模型
    model = YOLO(r'D:\workingSpace\Projects\train240502\runs\detect\train\weights\best.pt')
    camera_manager = CameraManager(4, (640, 480))  # 设置分辨率为640x480
    img_shape = (camera_manager.width, camera_manager.height)
    frame_processor = FrameProcessor(model, img_shape)
    print("*****YOLOv8检测模型加载成功*****\n")
    # 顺序处理每个相机
    while True:
        annotated_frames, azimuth = frame_processor.process_multiple_cameras(camera_manager.caps, CAMERA_ANGLES)
        frame_processor.display_multiple_cameras(annotated_frames,
                                                 (frame_processor.img_height, frame_processor.img_width),
                                                 "Cameras", (2, 2))
        if azimuth is not None:
            print(f"目标方位角：{np.round(azimuth, 2)}度")
        frame_processor.clear_azimuth_buffer()  # 每循环50次清空方位角缓存区
        # 等待按键输入，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭所有相机
    camera_manager.turn_off_cameras()
    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
