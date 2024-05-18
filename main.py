import time

import cv2
from ultralytics import YOLO

import camera
import processing
import visualization


# 主函数
def main():
    # 加载YOLOv8模型
    model = YOLO(r'D:\workingSpace\Projects\train240502\runs\detect\train\weights\best.pt')
    camera_manager = camera.CameraManager(num_cameras=4)
    frame_processor = processing.FrameProcessor(model)
    # 初始化时间
    prev_time = time.time()
    # 顺序处理每个相机
    while True:
        annotated_frames = frame_processor.run_processing(camera_manager.caps, camera.CAMERA_ANGLES)
        visualization.display_multiple_cameras(annotated_frames, (frame_processor.height, frame_processor.width),
                                               "Cameras", (2, 2))
        # 等待按键输入，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 关闭所有相机
    camera_manager.turn_off_cameras()
    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
