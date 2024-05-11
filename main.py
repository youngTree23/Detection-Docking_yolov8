import cv2


def main():
    # 打开四个相机
    num_cameras = 3
    caps = [cv2.VideoCapture(i+1) for i in range(num_cameras)]

    # 设置相机分辨率为1920x1080
    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 检查相机是否成功打开
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"无法打开相机 {i}")
            return

    while True:
        # 逐个相机捕获视频并显示
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取相机 {i} 的视频帧")
                continue

            # 缩小图像以适应屏幕大小
            resized_frame = cv2.resize(frame, (480, 270))

            # 显示缩小后的图像
            cv2.imshow(f'Camera {i}', resized_frame)

        # 检查是否按下了'q'键，如果是则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机对象
    for cap in caps:
        cap.release()

    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
