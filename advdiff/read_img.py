import numpy as np
from PIL import Image
from transformers import pipeline

# 读取npz文件
data = np.load('AdvDiff.npz')

# 假设图像数据存储在名为'arr_0'的数组中
img_data = data['arr_0']

# 确保图像数据在正确的范围内，通常是0-255
img_data = np.clip(img_data, 0, 255)

# 转换为图像
img = Image.fromarray((img_data[0] * 255).astype(np.uint8))
img = Image.fromarray((img_data[2] * 255).astype(np.uint8))
# 保存图像
img.save('image.png')