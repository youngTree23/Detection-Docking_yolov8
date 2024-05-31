import cv2

# 定义相机的中轴角度
CAMERA_ANGLES = [0, 90, 180, 270]
FOCAL_LENGTH_MM = 2.2
SENSOR_WIDTH_MM = 18.44
ANGLE_SCOPE = 135  # 定义相机识别的范围为135度


class CameraManager:
    def __init__(self, num_cameras, img_shape):
        """
        初始化相机管理器。

        参数:
            num_cameras (int): 要初始化的相机数量。
            width (int): 相机的帧宽度。
            height (int): 相机的帧高度。
        """
        self.num_cameras = num_cameras
        self.width, self.height = img_shape
        self.caps, self.errors = self.turn_on_cameras()
        if self.errors:
            print(f"以下相机未能成功打开: {self.errors}")
        else:
            print("*****相机已全部开启*****\n")

    def turn_on_cameras(self):
        """
        初始化多个相机，并设置分辨率。

        返回:
            caps (list): 包含所有成功打开的相机对象的列表。
            errors (list): 包含未能成功打开的相机索引的列表。
        """
        caps = [cv2.VideoCapture(0), cv2.VideoCapture(2), cv2.VideoCapture(3), cv2.VideoCapture(4)]

        errors = []

        for cap in caps:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        for i, cap in enumerate(caps, start=1):
            if not cap.isOpened():
                errors.append(i)

        return caps, errors

    def turn_off_cameras(self):
        """关闭所有相机"""
        for cap in self.caps:
            cap.release()


# 示例使用
if __name__ == "__main__":
    camera_manager = CameraManager(num_cameras=4, img_shape=(640, 480))

    # 关闭所有相机
    camera_manager.turn_off_cameras()
