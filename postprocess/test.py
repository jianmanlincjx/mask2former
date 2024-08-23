import cv2
import numpy as np

# 加载图像
image = cv2.imread('/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray_image, 50, 150)

# 检测角点（转折点）
corners = cv2.goodFeaturesToTrack(edges, maxCorners=50, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# 绘制转折点
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# 检测颜色变化点
color_change_points = []
for y in range(1, image.shape[0]-1):
    for x in range(1, image.shape[1]-1):
        if edges[y, x] > 0:
            color_diff = np.linalg.norm(image[y, x] - image[y, x-1])
            if color_diff > 50:  # 自定义阈值
                color_change_points.append((x, y))
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

# 保存结果
cv2.imwrite('/data1/JM/code/mask2former/postprocess/polygon_with_points.png', image)
