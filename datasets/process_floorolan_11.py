import os
from PIL import Image
import numpy as np


floorplan_fuse_map =   {
    # wall 白色
    1: (255, 255, 255),   
    # door 绿色, double-door 墨绿色, sliding-door 深红色
    2: [(0, 255, 0), (0, 126, 0), (126, 0, 0)],  
    # window 红色, balcony-window 棕色
    3: [(255, 0, 0), (126, 77, 7)],  
    # bay-window 粉红色, 
    4: (255, 191, 186), 
    # livingroom 浅蓝色, doorway 深蓝色
    5: [(3, 152, 255), (23, 0, 255), (251, 96, 255)],  
    # bedroom 橙色
    6: (254, 153, 3), 
    # kitchen 黄色    
    7: (0, 255, 255),    
    # bathroom 灰色 
    8: (127, 127, 127), 
    # library 肤色  
    9: (215, 210, 210), 
    # balcony 灰绿色     
    10: (98, 126, 91),  
    # other 黑色  
    11: (0, 0, 0),         
}


# Invert the floorplan_fuse_map to create a mapping from color to label
color_to_label = {}
for k, v in floorplan_fuse_map.items():
    if isinstance(v, list):
        for color in v:
            color_to_label[tuple(color)] = k
    else:
        color_to_label[tuple(v)] = k

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

    for color, label in color_to_label.items():
        mask = np.all(reshaped_img == color, axis=1)
        label_img[mask] = label

    label_img = label_img.reshape(img_np.shape[0], img_np.shape[1])

    output_img = Image.fromarray(label_img)
    output_img.save(os.path.join(output_dir, img_file))
    print(f"Processed {img_file}")

if __name__ == "__main__":
    image_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_black_w_color_1179/annotations/training_color"
    output_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_black_w_color_1179/annotations/training"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    colors = np.array(list(color_to_label.keys()))
    labels = np.array(list(color_to_label.values()))

    png_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    # 使用多运行过程处理
    with ProcessPoolExecutor() as executor:
        executor.map(process_image, png_files)

    print("All images processed successfully.")


# sudo mount -t cifs //192.168.0.180/Customer data ./Customer_data -o username=linjianman
