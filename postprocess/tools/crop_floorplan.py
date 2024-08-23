from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
import csv
import cv2
import math
import os

class CSVPointProcessor:
    def __init__(self, input_csv_path, output_csv_path, distance_threshold=1.5):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.distance_threshold = distance_threshold
        self.points = []
        self.filtered_points = []

    def load_csv(self):
        with open(self.input_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image = row['Image']
                position = eval(row['Position'])  # 将字符串的坐标转换为元组
                color = tuple(map(int, row['Color'].split(',')))  # 将颜色转换为元组
                self.points.append({'Image': image, 'Position': position, 'Color': color})

    # def filter_points(self):
    #     for i, point1 in enumerate(self.points):
    #         keep_point1 = True
    #         for j, point2 in enumerate(self.points):
    #             if i != j:
    #                 dist = math.dist(point1['Position'], point2['Position'])
    #                 if dist < self.distance_threshold:
    #                     if point1['Color'] == (128, 60, 255) and point2['Color'] != (128, 60, 255):
    #                         continue  # 保留 point1
    #                     elif point2['Color'] == (128, 60, 255) and point1['Color'] != (128, 60, 255):
    #                         keep_point1 = False  # 删除 point1
    #                         break
    #         if keep_point1:
    #             self.filtered_points.append(point1)

    def filter_points(self):
        for i, point1 in enumerate(self.points):
            keep_point1 = True
            for j, point2 in enumerate(self.points):
                if i != j:
                    # 1. 检查颜色是否相同
                    if point1['Color'] == point2['Color']:
                        continue  # 颜色相同，跳过对比
                    
                    # 2. 检查是否共线
                    if point1['Position'][0] != point2['Position'][0] and point1['Position'][1] != point2['Position'][1]:
                        continue  # x 或 y 值不同，不共线，跳过对比
                    
                    # 3. 计算距离并应用原有逻辑
                    dist = math.dist(point1['Position'], point2['Position'])
                    if dist < self.distance_threshold:
                        if point1['Color'] == (128, 60, 255) and point2['Color'] != (128, 60, 255):
                            continue  # 保留 point1
                        elif point2['Color'] == (128, 60, 255) and point1['Color'] != (128, 60, 255):
                            keep_point1 = False  # 删除 point1
                            break
            if keep_point1:
                self.filtered_points.append(point1)

    def save_filtered_csv(self):
        with open(self.output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Position', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in self.filtered_points:
                writer.writerow({
                    'Image': point['Image'],
                    'Position': f'({point["Position"][0]},{point["Position"][1]})',
                    'Color': f'{point["Color"][0]},{point["Color"][1]},{point["Color"][2]}'
                })
        print(f"Filtered CSV saved at {self.output_csv_path}")

    def process(self):
        self.load_csv()
        self.filter_points()
        self.save_filtered_csv()


class ImageAnnotator:
    def __init__(self, image_path, csv_path, output_image_path):
        self.image_path = image_path
        self.csv_path = csv_path
        self.output_image_path = output_image_path

    def draw_points_on_image(self):
        image = cv2.imread(self.image_path)

        with open(self.csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                position = eval(row['Position'])  # 将字符串的坐标转换为元组
                color = tuple(map(int, row['Color'].split(',')))  # 将颜色转换为元组
                # 绘制点在图像上
                cv2.circle(image, position, radius=5, color=color, thickness=-1)

        flipped_image_180 = cv2.rotate(image, cv2.ROTATE_180)
        flipped_image_final = cv2.flip(flipped_image_180, 1)
        cv2.imwrite(self.output_image_path, flipped_image_final)

        print(f"Annotated image saved at {self.output_image_path}")

class FloorplanExtractor:
    def __init__(self, image_path, output_image_path1, output_image_path2, csv_output_path):
        self.image_path = image_path
        self.output_image_path1 = output_image_path1
        self.output_image_path2 = output_image_path2
        self.csv_output_path = csv_output_path

    def extract_subimages(self):
        # 读取输入图像
        image = cv2.imread(self.image_path)

        # 定义颜色范围（下限和上限）
        color1 = np.array([128, 60, 255])
        color2 = np.array([255, 255, 255])
        background = np.array([0, 0, 0])

        # 创建两个空白图像，用于存储提取的部分
        extracted_image1 = np.zeros_like(image)
        extracted_image2 = np.zeros_like(image)

        # 创建掩码，用于提取指定颜色的部分
        mask1 = cv2.inRange(image, color1, color1)
        mask2 = cv2.inRange(image, color2, color2)
        mask_bg = cv2.inRange(image, background, background)

        # 提取第一个子图，包含 color1 和背景
        extracted_image1[mask1 > 0] = color1
        extracted_image1[mask_bg > 0] = background

        # 提取第二个子图，包含 color2 和背景
        extracted_image2[mask2 > 0] = color2
        extracted_image2[mask_bg > 0] = background

        flipped_image1_180 = cv2.rotate(extracted_image1, cv2.ROTATE_180)
        flipped_image1_final = cv2.flip(flipped_image1_180, 1)

        # 对 extracted_image2 进行180°翻转，然后左右翻转
        flipped_image2_180 = cv2.rotate(extracted_image2, cv2.ROTATE_180)
        flipped_image2_final = cv2.flip(flipped_image2_180, 1)

        # 保存翻转后的图像
        cv2.imwrite(self.output_image_path1, flipped_image1_final)
        cv2.imwrite(self.output_image_path2, flipped_image2_final)

        self._detect_and_save_corners(extracted_image1, extracted_image2)

    def _detect_and_save_corners(self, image1, image2):
        # 打开CSV文件以写入角点信息
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Position', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # 对每个子图单独进行角点检测并记录
            for idx, (sub_image, img_label) in enumerate(zip([image1, image2], ["windows", "wall"])):
                gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)

                # 调整角点检测参数
                corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=200, qualityLevel=0.01, minDistance=2.5)
                if corners is not None:
                    corners = np.int0(corners)

                    for corner in corners:
                        x, y = corner.ravel()
                        color = tuple(sub_image[y, x])  # 获取角点的颜色
                        # 过滤掉颜色为 (0, 0, 0) 的点
                        if color != (0, 0, 0):
                            writer.writerow({'Image': img_label, 'Position': f'({x},{y})', 'Color': f'{color[0]},{color[1]},{color[2]}'})

    def run(self):
        self.extract_subimages()

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
        dilated_mask = binary_dilation(livingroom_mask, structure=np.ones((7, 7))).astype(np.uint8)

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
    image_path = '/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/45741118.png'


    # 提取房间区域及其边界的子户型图
    processor = FloorplanProcessor(
        image_path=image_path,
        output_path="/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png",
        padding=20
    )
    processor.process_image()

    image = Image.open(image_path)
    rotated_image = image.rotate(180)
    flipped_image = rotated_image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image_path = '/data1/JM/code/mask2former/postprocess/result/image.png'
    flipped_image.save(flipped_image_path)


    # 对提取的子户型图进行角点检测
    extractor = FloorplanExtractor(
        image_path='/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png',
        output_image_path1='/data1/JM/code/mask2former/postprocess/result/floorplan_windows.png',
        output_image_path2='/data1/JM/code/mask2former/postprocess/result/floorplan_wall.png',
        csv_output_path='/data1/JM/code/mask2former/postprocess/result/corners_info.csv'
    )
    extractor.run()

    # 多余角点过滤
    processor = CSVPointProcessor(
        input_csv_path='/data1/JM/code/mask2former/postprocess/result/corners_info.csv',
        output_csv_path='/data1/JM/code/mask2former/postprocess/result/filtered_corners_info.csv',
        distance_threshold=5
    )
    processor.process()

    # 角点贴回原图debug
    annotator = ImageAnnotator(
        image_path='/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png',
        csv_path='/data1/JM/code/mask2former/postprocess/result/filtered_corners_info.csv',
        output_image_path='/data1/JM/code/mask2former/postprocess/result/annotated_output_image.png'
    )
    annotator.draw_points_on_image()