import numpy as np
def add_sunlight(video_data):
    video_data = video_data.astype(np.uint8)
    # 获取视频的帧数、宽度和高度
    frames, rows, cols, channels = video_data.shape

    # 设置中心点
    centerX = rows / 2
    centerY = cols / 2
    radius = min(centerX, centerY)

    # 设置光照强度
    # 定义可能的数值
    values = [0, 10, 20, 30, 40, 50]
    # 定义每个数值的概率
    probabilities = [0.25, 0.15, 0.15, 0.15, 0.15, 0.15]

    # 生成一个随机数
    def generate_random_number():
        return np.random.choice(values, p=probabilities)

    strength = generate_random_number()
    if strength==0:
        return video_data, strength
    # print(strength)
    c_point = np.array([centerX, centerY])

    # 创建网格
    y = np.arange(0, rows)
    x = np.arange(0, cols)
    res = np.meshgrid(x, y)
    grid = np.concatenate((np.expand_dims(res[1], -1), np.expand_dims(res[0], -1)), axis=2)

    # 计算每个点到中心点的距离
    distance = np.linalg.norm(c_point - grid, axis=2)

    # 计算光照增强
    light = (strength * (1 - distance / radius))
    light = np.where(light > 0, light, 0).astype(np.int8)

    # 将光照增强应用到每一帧
    for i in range(frames):
        frame = video_data[i]
        frame = np.clip(frame + np.expand_dims(light, -1), 0, 255)
        video_data[i] = frame
    return video_data, strength