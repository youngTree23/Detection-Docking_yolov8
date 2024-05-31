def check_angle(angle, threshold=5):
    """
    判断当前方位角是否在90度附近，即判断1号相机是否正对目标物体
    :param angle:
    :param threshold:
    :return:
    """
    lower_bound = 0 - threshold
    upper_bound = 0 + threshold

    if lower_bound <= angle <= upper_bound:
        return 1
    else:
        return 0
