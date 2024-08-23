from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
import csv
import cv2
import math
import os
import pandas as pd


class CornerDetectorGFTT:
    def __init__(self, image_path, output_csv_path, output_image_path):
        self.image_path = image_path
        self.output_csv_path = output_csv_path
        self.output_image_path = output_image_path
        self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        color_to_replace = np.array([224, 255, 192])
        black_color = np.array([0, 0, 0])
        mask = np.all(self.image == color_to_replace, axis=-1)
        self.image[mask] = black_color

        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def detect_corners(self, max_corners=200, quality_level=0.01, min_distance=2.5):
        # Detect corners using Shi-Tomasi method (Good Features to Track)
        corners = cv2.goodFeaturesToTrack(self.gray_image, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
        self.corners = np.int0(corners)
        return self.corners

    def check_surrounding_colors(self, x, y):
        colors = set()
        # Checking left/right and up/down within a range of 1 pixel
        for dx in range(-1, 2):  # x-direction: left 1 pixel, right 1 pixel
            for dy in range(-1, 2):  # y-direction: up 1 pixel, down 1 pixel
                if dx == 0 and dy == 0:
                    continue  # Skip the center point itself
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.image.shape[1] and 0 <= ny < self.image.shape[0]:
                    color = tuple(self.image[ny, nx])
                    if color != (0, 0, 0) and color != [224, 255, 192]:  # Ignore black background color
                        colors.add(color)
        return colors

    def classify_and_save(self):
        output_data = []

        # Iterate over the detected corners
        for corner in self.corners:
            x, y = corner.ravel()
            colors = self.check_surrounding_colors(x, y)
            color_str = ','.join(map(str, self.image[y, x]))

            if len(colors) > 1:
                category = 'windows'
                cv2.circle(self.image, (x, y), 5, (0, 255, 0), -1)  # Mark windows with green
            else:
                category = 'wall'
                cv2.circle(self.image, (x, y), 5, (255, 0, 0), -1)  # Mark walls with blue

            position_str = f"({x},{y})"
            output_data.append([category, position_str, color_str])

        # Save to CSV
        df = pd.DataFrame(output_data, columns=['Type', 'Position', 'Color'])
        df.to_csv(self.output_csv_path, index=False)
        print(f"CSV saved to: {self.output_csv_path}")

        # Save the marked image
        cv2.imwrite(self.output_image_path, self.image)
        print(f"Image with marked corners saved to: {self.output_image_path}")

    def run(self):
        self.detect_corners()
        self.classify_and_save()



class ImageAnnotator:
    def __init__(self, image_path, csv_path, output_image_path):
        self.image_path = image_path
        self.csv_path = csv_path
        self.output_image_path = output_image_path

    def draw_points_on_image(self):
        # 读取原始图像
        image = cv2.imread(self.image_path)

        # 打开CSV文件并读取数据
        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                position = eval(row['Position'])  # 将字符串的坐标转换为元组
                color = tuple(map(int, row['Color'].split(',')))  # 将颜色转换为元组
                # 绘制点在图像上
                cv2.circle(image, position, radius=5, color=color, thickness=-1)

        # 进行180°翻转
        flipped_image_180 = cv2.rotate(image, cv2.ROTATE_180)

        # 进行左右翻转（水平翻转）
        flipped_image_final = cv2.flip(flipped_image_180, 1)

        # 保存最终翻转后的图像
        cv2.imwrite(self.output_image_path, flipped_image_final)

        print(f"Annotated image saved at {self.output_image_path}")


class FloorplanProcessor:
    def __init__(self, image_path, output_path, padding=20):
        self.image_path = image_path
        self.output_path = output_path
        self.padding = padding
        
        # 定义颜色映射
        self.floorplan_fuse_map = {
            0: [0, 0, 0],           # background
            1: [192, 192, 224],     # closet
            2: [192, 255, 255],     # batchroom/washroom
            3: [224, 255, 192],     # livingroom/kitchen/dining room
            4: [255, 224, 128],     # bedroom
            5: [255, 160, 96],      # hall
            6: [255, 224, 224],     # balcony
            7: [224, 224, 224],     # not used
            8: [224, 224, 128],     # not used
            9: [255, 60, 128],      # extra label for opening (door&window)
            10: [255, 255, 255]     # extra label for wall line
        }

    def process_image(self):
        # 读取图像并转换为 numpy 数组
        image = Image.open(self.image_path)
        image_np = np.array(image)

        # 创建掩模：找到 livingroom 区域
        livingroom_color = np.array(self.floorplan_fuse_map[3])
        livingroom_mask = np.all(image_np == livingroom_color, axis=-1)

        # 扩张区域以包含边界信息
        dilated_mask = binary_dilation(livingroom_mask, structure=np.ones((5, 5))).astype(np.uint8)

        # 创建一个新的图像来保存结果
        result_image = np.zeros_like(image_np)

        # 仅保留 livingroom 区域及扩展部分的颜色
        result_image[dilated_mask > 0] = image_np[dilated_mask > 0]

        # 过滤颜色，只保留 livingroom 区域及扩展部分的颜色
        keep_colors = [self.floorplan_fuse_map[3], self.floorplan_fuse_map[9], self.floorplan_fuse_map[10]]
        keep_colors = np.array(keep_colors)

        # 创建掩模用于过滤颜色
        color_mask = np.zeros_like(image_np, dtype=bool)
        for color in keep_colors:
            color_mask |= np.all(result_image == color, axis=-1)[:, :, np.newaxis]

        # 应用颜色掩模
        result_image[~color_mask] = 0

        # 计算非零区域的边界
        non_zero_indices = np.argwhere(np.any(result_image != 0, axis=-1))
        y_min, x_min = np.min(non_zero_indices, axis=0)
        y_max, x_max = np.max(non_zero_indices, axis=0)

        # 扩展边界以保留空隙
        y_min = max(y_min - self.padding, 0)
        x_min = max(x_min - self.padding, 0)
        y_max = min(y_max + self.padding, result_image.shape[0] - 1)
        x_max = min(x_max + self.padding, result_image.shape[1] - 1)

        # 裁剪图像
        cropped_image = result_image[y_min:y_max+1, x_min:x_max+1]

        # 保存裁剪后的结果图像
        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_image_pil.save(self.output_path)
        print(f"Processed image saved to {self.output_path}")



# Example usage
if __name__ == "__main__":
    os.system(f'rm -rf /data1/JM/code/mask2former/postprocess/result')
    os.makedirs('/data1/JM/code/mask2former/postprocess/result')
    image_path = '/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/45724345.png'
    
    output_crop_image_path = "/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png"
    output_csv_path = '/data1/JM/code/mask2former/postprocess/result/filtered_corners_info.csv'
    output_annotated_image_path = '/data1/JM/code/mask2former/postprocess/result/annotated_output_image.png'

    # 提取房间区域及其边界的子户型图
    processor = FloorplanProcessor(
        image_path=image_path,
        output_path=output_crop_image_path,
        padding=20
    )
    processor.process_image()

    image = Image.open(image_path)
    rotated_image = image.rotate(180)
    flipped_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image_path = '/data1/JM/code/mask2former/postprocess/result/image.png'
    flipped_image.save(flipped_image_path)

    detector = CornerDetectorGFTT(output_crop_image_path, output_csv_path, output_annotated_image_path)
    detector.run()

