import os
from PIL import Image
import numpy as np

floorplan_fuse_map = {
    1: (255, 255, 255),   # wall 墙壁 白色
    2: (0, 255, 0),       # door 门 绿色
    3: (0, 126, 0),       # double-door 双门 墨绿色
    4: (126, 0, 0),       # sliding-door 推拉门 深红色
    5: (255, 0, 0),       # window 窗户 红色
    6: (255, 191, 186),   # bay-window 飘窗 粉红色
    7: (126, 77, 7),      # balcony-window 阳台窗 棕色
    8: (23, 0, 255),      # doorway 门洞 深蓝色
    9: (3, 152, 255),     # livingroom 客厅 浅蓝色
    10: (254, 153, 3),    # bedroom 卧室 橙色
    11: (0, 255, 255),    # kitchen 厨房 黄色
    12: (127, 127, 127),  # bathroom 浴室 灰色
    13: (215, 210, 210),  # library 书房 肤色
    14: (98, 126, 91),    # balcony 阳台 灰绿色
    15: (251, 96, 255),   # diningroom 餐厅 紫色
    16: (0, 0, 0),        # other 其他 黑色
}



# Invert the floorplan_fuse_map to create a mapping from color to label
color_to_label = {tuple(v): k for k, v in floorplan_fuse_map.items()}

import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def process_image(img_file):
    img_path = os.path.join(image_dir, img_file)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    reshaped_img = img_np.reshape(-1, 3)
    label_img = np.zeros(reshaped_img.shape[0], dtype=np.uint8)

    for i, color in enumerate(colors):
        mask = np.all(reshaped_img == color, axis=1)
        label_img[mask] = labels[i]

    label_img = label_img.reshape(img_np.shape[0], img_np.shape[1])

    output_img = Image.fromarray(label_img)
    output_img.save(os.path.join(output_dir, img_file))
    print(f"Processed {img_file}")

if __name__ == "__main__":
    image_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586/annotations/validation_color"
    output_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586/annotations/validation"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colors = np.array(list(color_to_label.keys()))
    labels = np.array(list(color_to_label.values()))

    png_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    # 使用多进程处理
    with ProcessPoolExecutor() as executor:
        executor.map(process_image, png_files)

    print("All images processed successfully.")


# sudo mount -t cifs //192.168.0.180/Customer data ./Customer_data -o username=linjianman