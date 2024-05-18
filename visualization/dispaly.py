import cv2
import numpy as np


def display_single_camera(frame, index):
    cv2.imshow(f'Camera {index}', frame)


def display_multiple_cameras(frames, shape, window_name="Cameras", grid_size=(2, 2)):
    """
    显示多个相机的画面在一个窗口中。
    """
    rows, cols = grid_size
    height, width = shape

    # 将帧组合成一个网格图像
    grid_image = np.zeros((height * rows, width * cols, 3), np.uint8)
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        grid_image[row * height:(row + 1) * height, col * width:(col + 1) * width] = frame

    cv2.imshow(window_name, grid_image)

