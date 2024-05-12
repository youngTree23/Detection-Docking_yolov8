import cv2


def main():
    num_cameras = 4  # 假设最多有8个相机连接到计算机上

    for i in range(1, num_cameras + 1):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            print(f"相机 {i} 可用")
            cap.release()
        else:
            print(f"相机 {i} 不可用")


if __name__ == "__main__":
    main()
