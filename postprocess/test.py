import cv2
import numpy as np

# 读取分割图像
segmentation_map = cv2.imread('segmentation_map.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if segmentation_map is None:
    raise ValueError("图像读取失败，请检查文件路径。")

# 生成热图
heatmap = cv2.applyColorMap(segmentation_map, cv2.COLORMAP_JET)

# 获取图像的尺寸
height, width = segmentation_map.shape

# 输出原始灰度值与热图RGB值的对应关系
for i in range(height):
    for j in range(width):
        grayscale_value = segmentation_map[i, j]  # 原始灰度值
        heatmap_rgb = heatmap[i, j]  # 对应的热图RGB值
        
        # 输出对应关系
        print(f"位置 ({i}, {j}): 灰度值 = {grayscale_value}, 热图 RGB 值 = {heatmap_rgb}")

# 如果不想输出所有像素的对应关系，可以将上面的遍历范围进行缩小，比如只输出部分像素的对应关系。
