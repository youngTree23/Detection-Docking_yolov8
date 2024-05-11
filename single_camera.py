import cv2


def main():
    # 打开默认相机
    cap = cv2.VideoCapture(1)

    # 设置相机分辨率为1920x1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # 检查相机是否成功打开
    if not cap.isOpened():
        print("无法打开相机")
        return

    while True:
        # 逐帧捕获视频
        ret, frame = cap.read()

        # 检查帧是否成功读取
        if not ret:
            print("无法读取视频帧")
            break

        # 缩小图像以适应屏幕大小
        resized_frame = cv2.resize(frame, (960, 540))

        # 显示缩小后的图像
        cv2.imshow('Camera', resized_frame)

        # 检查是否按下了'q'键，如果是则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放相机对象
    cap.release()

    # 关闭所有打开的窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
