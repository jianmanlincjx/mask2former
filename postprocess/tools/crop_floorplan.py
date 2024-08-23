from PIL import Image
import numpy as np
from scipy.ndimage import binary_dilation
import csv
import cv2
import math
import os
import pandas as pd

class FloorplanProcessingPipeline:
    def __init__(self, image_path, output_dir, padding=20, distance_threshold=1.5, window_distance_threshold=4):
        self.image_path = image_path
        self.output_dir = output_dir
        self.padding = padding
        self.distance_threshold = distance_threshold
        self.window_distance_threshold = window_distance_threshold
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
        if os.path.exists(self.output_dir):
            os.system(f'rm -rf {self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)

    def process_image(self):
        image = Image.open(self.image_path)
        image_np = np.array(image)
        livingroom_color = np.array(self.floorplan_fuse_map[3])
        livingroom_mask = np.all(image_np == livingroom_color, axis=-1)
        dilated_mask = binary_dilation(livingroom_mask, structure=np.ones((7, 7))).astype(np.uint8)
        result_image = np.zeros_like(image_np)
        result_image[dilated_mask > 0] = image_np[dilated_mask > 0]

        keep_colors = [self.floorplan_fuse_map[3], self.floorplan_fuse_map[9], self.floorplan_fuse_map[10]]
        keep_colors = np.array(keep_colors)

        color_mask = np.zeros_like(image_np, dtype=bool)
        for color in keep_colors:
            color_mask |= np.all(result_image == color, axis=-1)[:, :, np.newaxis]

        result_image[~color_mask] = 0
        non_zero_indices = np.argwhere(np.any(result_image != 0, axis=-1))
        y_min, x_min = np.min(non_zero_indices, axis=0)
        y_max, x_max = np.max(non_zero_indices, axis=0)

        y_min = max(y_min - self.padding, 0)
        x_min = max(x_min - self.padding, 0)
        y_max = min(y_max + self.padding, result_image.shape[0] - 1)
        x_max = min(x_max + self.padding, result_image.shape[1] - 1)

        cropped_image = result_image[y_min:y_max+1, x_min:x_max+1]

        cropped_image_pil = Image.fromarray(cropped_image)
        cropped_output_path = os.path.join(self.output_dir, "cropped_output_image.png")
        cropped_image_pil.save(cropped_output_path)

        print(f"Processed image saved to {cropped_output_path}")
        return cropped_output_path

    def extract_subimages(self, cropped_image_path):
        image = cv2.imread(cropped_image_path)

        color1 = np.array([128, 60, 255])
        color2 = np.array([255, 255, 255])
        background = np.array([0, 0, 0])

        extracted_image1 = np.zeros_like(image)
        extracted_image2 = np.zeros_like(image)

        mask1 = cv2.inRange(image, color1, color1)
        mask2 = cv2.inRange(image, color2, color2)
        mask_bg = cv2.inRange(image, background, background)

        extracted_image1[mask1 > 0] = color1
        extracted_image1[mask_bg > 0] = background
        extracted_image2[mask2 > 0] = color2
        extracted_image2[mask_bg > 0] = background

        output_image_path1 = os.path.join(self.output_dir, "floorplan_windows.png")
        output_image_path2 = os.path.join(self.output_dir, "floorplan_wall.png")
        cv2.imwrite(output_image_path1, extracted_image1)
        cv2.imwrite(output_image_path2, extracted_image2)

        csv_output_path = os.path.join(self.output_dir, "corners_info.csv")
        self._detect_and_save_corners(extracted_image1, extracted_image2, csv_output_path)

        print(f"Extracted subimages and saved corners to {csv_output_path}")
        return csv_output_path

    def _detect_and_save_corners(self, image1, image2, csv_output_path):
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Position', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for sub_image, img_label in zip([image1, image2], ["windows", "wall"]):
                gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
                corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=1000, qualityLevel=0.01, minDistance=1)
                if corners is not None:
                    corners = np.int0(corners)
                    for corner in corners:
                        x, y = corner.ravel()
                        color = tuple(sub_image[y, x])
                        if color != (0, 0, 0):
                            writer.writerow({'Image': img_label, 'Position': f'({x},{y})', 'Color': f'{color[0]},{color[1]},{color[2]}'})

    def filter_points(self, csv_input_path):
        points = []
        filtered_points = []

        with open(csv_input_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image = row['Image']
                position = eval(row['Position'])
                color = tuple(map(int, row['Color'].split(',')))
                points.append({'Image': image, 'Position': position, 'Color': color})

        for i, point1 in enumerate(points):
            keep_point1 = True
            for j, point2 in enumerate(points):
                if i != j:
                    if point1['Color'] == point2['Color']:
                        continue
                    
                    dist = math.dist(point1['Position'], point2['Position'])
                    if dist < self.distance_threshold:
                        if point1['Color'] == (128, 60, 255) and point2['Color'] != (128, 60, 255):
                            continue
                        elif point2['Color'] == (128, 60, 255) and point1['Color'] != (128, 60, 255):
                            keep_point1 = False
                            break
            if keep_point1:
                filtered_points.append(point1)

        filtered_csv_output_path = os.path.join(self.output_dir, "filtered_corners_info.csv")
        with open(filtered_csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['Image', 'Position', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in filtered_points:
                writer.writerow({
                    'Image': point['Image'],
                    'Position': f'({point["Position"][0]},{point["Position"][1]})',
                    'Color': f'{point["Color"][0]},{point["Color"][1]},{point["Color"][2]}'
                })

        print(f"Filtered points saved to {filtered_csv_output_path}")
        return filtered_csv_output_path

    def filter_close_windows(self, csv_input_path):
        def calculate_distance(point1, point2):
            return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

        df = pd.read_csv(csv_input_path)
        df['Position'] = df['Position'].apply(lambda pos: eval(pos))
        windows_df = df[df['Image'] == 'windows']

        to_remove = set()
        for i in range(len(windows_df)):
            for j in range(i + 1, len(windows_df)):
                pos1 = windows_df.iloc[i]['Position']
                pos2 = windows_df.iloc[j]['Position']
                if calculate_distance(pos1, pos2) <= self.window_distance_threshold:
                    to_remove.add(windows_df.index[j])

        filtered_df = df.drop(to_remove)
        output_csv_path = os.path.join(self.output_dir, "filtered_corners_info_close_windows.csv")
        filtered_df.to_csv(output_csv_path, index=False)

        print(f"Filtered close windows saved to {output_csv_path}")
        return output_csv_path

    def annotate_image(self, filtered_csv_path, cropped_image_path, save_name):
        image = cv2.imread(cropped_image_path)

        with open(filtered_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                position = eval(row['Position'])
                color = tuple(map(int, row['Color'].split(',')))
                cv2.circle(image, position, radius=5, color=color, thickness=-1)

        # Save the original annotated image
        output_image_path = os.path.join(self.output_dir, save_name)
        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved to {output_image_path}")

        # 放大图像
        scale_factor = 2  # 放大的倍数，可以根据需要调整
        enlarged_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

        with open(filtered_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # 根据放大倍数调整坐标
                position = eval(row['Position'])
                enlarged_position = (int(position[0] * scale_factor), int(position[1] * scale_factor))
                color = tuple(map(int, row['Color'].split(',')))

                # 在放大的图像上绘制圆点
                cv2.circle(enlarged_image, enlarged_position, radius=5, color=color, thickness=-1)

                # 在放大的图像上添加坐标文本
                text_position = (enlarged_position[0] + 10, enlarged_position[1] - 10)
                cv2.putText(enlarged_image, f"({position[0]}, {position[1]})", text_position, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # 保存放大并标注后的图像
        output_image_with_text_path = os.path.join(self.output_dir, "annotated_output_image_with_text_large.png")
        cv2.imwrite(output_image_with_text_path, enlarged_image)

    def run(self):
        cropped_image_path = self.process_image()
        csv_output_path = self.extract_subimages(cropped_image_path)
        self.annotate_image(csv_output_path, cropped_image_path, save_name='filter-before_annotate_image.png')
        filtered_csv_path = self.filter_points(csv_output_path)
        filtered_close_windows_csv = self.filter_close_windows(filtered_csv_path)
        self.annotate_image(filtered_close_windows_csv, cropped_image_path, save_name='annotated_output_image.png')

# Example usage
if __name__ == "__main__":
    pipeline = FloorplanProcessingPipeline(
        image_path="/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/45724345.png",
        output_dir="/data1/JM/code/mask2former/postprocess/result",
        padding=20,
        distance_threshold=5,
        window_distance_threshold=2
    )
    pipeline.run()


#     # image_path = '/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/45783298.png'
#     image_path = '/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/31830006.png'
#     # image_path = '/data1/JM/code/mask2former/datasets/FloorPlan/annotations/training_original/45720007.png'