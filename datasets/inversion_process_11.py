import os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

# 定义 floorplan_fuse_map，映射标签编号到颜色
floorplan_fuse_map = {
    1: (255, 255, 255),   
    2: (0, 255, 0) ,
    3: (255, 0, 0), 
    4: (255, 191, 186), 
    5: (3, 152, 255), 
    6: (254, 153, 3), 
    7: (0, 255, 255),    
    8: (127, 127, 127), 
    9: (215, 210, 210), 
    10: (98, 126, 91),  
    11: (0, 0, 0),         
}


# 从 floorplan_fuse_map 中生成 label_to_color
label_to_color = {k: v for k, v in floorplan_fuse_map.items()}

def inverse_process_image(label_file):
    label_path = os.path.join(label_image_dir, label_file)
    label_img = Image.open(label_path).convert("L")
    label_np = np.array(label_img)

    # 创建一个新的 RGB 图像，大小和标签图像相同
    color_img_np = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)

    # 将标签值映射回颜色值
    for label, color in label_to_color.items():
        mask = label_np == label
        color_img_np[mask] = color

    # 保存为 RGB 图像
    color_img = Image.fromarray(color_img_np)
    output_path = os.path.join(color_output_dir, label_file)
    color_img.save(output_path)
    print(f"Inverse processed {label_file} and saved to {output_path}")

if __name__ == "__main__":
    label_image_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11/annotations/validation"
    color_output_dir = "/data1/JM/code/mask2former/post_process_data/test_data/label"

    if not os.path.exists(color_output_dir):
        os.makedirs(color_output_dir)

    # 获取所有的标签图像文件
    label_files = [f for f in os.listdir(label_image_dir) if f.endswith(".png")]

    # 使用多进程处理
    with ProcessPoolExecutor() as executor:
        executor.map(inverse_process_image, label_files)

    print("All label images converted back to color successfully.")