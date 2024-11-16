import os
from PIL import Image

# 文件夹路径
folder1 = "/data1/JM/code/mask2former/金牌测试/input/all"
folder2 = "/data1/JM/code/mask2former/金牌测试/result/all_139999_postprocess"
output_folder = "/data1/JM/code/mask2former/金牌测试/result_concat_postprocess"

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历第一个文件夹中的文件
for filename in os.listdir(folder1):
    # 确保只处理图像文件
    if filename.endswith(".png") or filename.endswith(".jpg"):
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        # 检查第二个文件夹中是否有相同的文件名
        if os.path.exists(path2):
            # 打开两个图像
            image1 = Image.open(path1)
            image2 = Image.open(path2)

            # 确保两个图像大小相同
            if image1.size == image2.size:
                # 水平拼接图像
                combined_image = Image.new('RGB', (image1.width + image2.width, image1.height))
                combined_image.paste(image1, (0, 0))
                combined_image.paste(image2, (image1.width, 0))

                # 保存拼接后的图像
                combined_image.save(os.path.join(output_folder, filename))
            else:
                print(f"Image sizes do not match for {filename}, skipping...")
        else:
            print(f"Matching file for {filename} not found in folder 2, skipping...")






# import os
# import numpy as np
# from PIL import Image
# from concurrent.futures import ProcessPoolExecutor

# # 定义 floorplan_fuse_map，映射标签编号到颜色
# floorplan_fuse_map = {
#     1: (255, 255, 255),   # wall 白色
#     2: (0, 255, 0),       # door 绿色
#     3: (0, 126, 0),       # double-door 墨绿色 
#     4: (126, 0, 0),       # sliding-door 深红色 
#     5: (255, 0, 0),       # window 红色
#     6: (255, 191, 186),   # bay-window 粉红色
#     7: (126, 77, 7),      # balcony-window 棕色 
#     8: (23, 0, 255),      # doorway 深蓝色
#     9: (3, 152, 255),     # livingroom 浅蓝色
#     10: (254, 153, 3),    # bedroom 橙色
#     11: (0, 255, 255),    # kitchen 黄色
#     12: (127, 127, 127),  # bathroom 灰色
#     13: (215, 210, 210),  # library 肤色
#     14: (98, 126, 91),    # balcony 灰绿色        
#     15: (251, 96, 255),   # diningroom 紫色
#     16: (0, 0, 0),        # other 黑色
# }

# # 从 floorplan_fuse_map 中生成 label_to_color
# label_to_color = {k: v for k, v in floorplan_fuse_map.items()}

# def inverse_process_image(label_file):
#     label_path = os.path.join(label_image_dir, label_file)
#     label_img = Image.open(label_path).convert("L")
#     label_np = np.array(label_img)

#     # 创建一个新的 RGB 图像，大小和标签图像相同
#     color_img_np = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)

#     # 将标签值映射回颜色值
#     for label, color in label_to_color.items():
#         mask = label_np == label
#         color_img_np[mask] = color

#     # 保存为 RGB 图像
#     color_img = Image.fromarray(color_img_np)
#     output_path = os.path.join(color_output_dir, label_file)
#     color_img.save(output_path)
#     print(f"Inverse processed {label_file} and saved to {output_path}")

# if __name__ == "__main__":
#     label_image_dir = "/data1/JM/code/mask2former/datasets/annotation_dataset_586/annotations/training"
#     color_output_dir = "/data1/JM/code/mask2former/temp/train_label_color_images"

#     if not os.path.exists(color_output_dir):
#         os.makedirs(color_output_dir)

#     # 获取所有的标签图像文件
#     label_files = [f for f in os.listdir(label_image_dir) if f.endswith(".png")]

#     # 使用多进程处理
#     with ProcessPoolExecutor() as executor:
#         executor.map(inverse_process_image, label_files)

#     print("All label images converted back to color successfully.")
