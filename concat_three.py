import os
from PIL import Image

# 文件夹路径
image_folder_1 = "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_only_black/images/validation"
image_folder_2 = "/data1/JM/code/mask2former/金牌测试/result/Base_resnet50_only_black_1127"
image_folder_3 = "/data1/JM/code/mask2former/datasets/annotation_dataset_586_11_only_black/annotations/validation_color"

# 输出文件夹
output_folder = "/data1/JM/code/mask2former/金牌测试/result/merged_images_black_1127"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取三个文件夹中的图像文件列表
image_files_1 = sorted(os.listdir(image_folder_1))
image_files_2 = sorted(os.listdir(image_folder_2))
image_files_3 = sorted(os.listdir(image_folder_3))

# 假设三个文件夹的图像文件是匹配的，按顺序处理
for i in range(len(image_files_1)):
    img_path_1 = os.path.join(image_folder_1, image_files_1[i])
    img_path_2 = os.path.join(image_folder_2, image_files_2[i])
    img_path_3 = os.path.join(image_folder_3, image_files_3[i])

    # 打开图像
    img1 = Image.open(img_path_1)
    img2 = Image.open(img_path_2)
    img3 = Image.open(img_path_3)

    # 确保三个图像的高度相同，可以进行拼接
    height = max(img1.height, img2.height, img3.height)
    img1 = img1.resize((img1.width, height))
    img2 = img2.resize((img2.width, height))
    img3 = img3.resize((img3.width, height))

    # 中间留的缝隙宽度
    gap_width = 10  # 根据需要调整

    # 创建一个新的空白图像，用来拼接
    total_width = img1.width + img2.width + img3.width + 2 * gap_width
    new_image = Image.new('RGB', (total_width, height), (255, 255, 255))

    # 将三张图像拼接到新图像中
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (img1.width + gap_width, 0))
    new_image.paste(img3, (img1.width + img2.width + 2 * gap_width, 0))

    # 保存拼接后的图像
    output_path = os.path.join(output_folder, f"merged_{i+1}.jpg")
    new_image.save(output_path)

    print(f"Image {i+1} merged and saved as {output_path}")



# import os
# import numpy as np
# import cv2

# # 定义路径
# test_edge_type_dir = "/data1/JM/code/mask2former/金牌测试/result/Base_resnet50_split/Base_resnet101_11_only_color_edge"
# validation_set_dir = "/data1/JM/code/mask2former/金牌测试/result/Base_resnet50_split/Base_resnet101_11_only_color_room"
# output_dir = "/data1/JM/code/mask2former/金牌测试/result/Base_resnet50_split/concat"

# # 创建输出文件夹（如果不存在）
# os.makedirs(output_dir, exist_ok=True)

# # 设置误差阈值
# threshold = 30

# # 遍历 test_edge_type 目录中的文件
# for filename in os.listdir(test_edge_type_dir):
#     # 只处理图像文件
#     if filename.endswith((".png", ".jpg", ".jpeg")):
#         test_edge_path = os.path.join(test_edge_type_dir, filename)
#         validation_path = os.path.join(validation_set_dir, filename)
#         output_path = os.path.join(output_dir, filename)
        
#         # 检查是否在 validation_set 目录中存在匹配的文件
#         if os.path.exists(validation_path):
#             # 使用 OpenCV 读取两张图像，不需要透明通道
#             edge_img = cv2.imread(test_edge_path)
#             validation_img = cv2.imread(validation_path)
            
#             # 创建掩码（mask），选择 RGB 值接近 (0, 0, 0) 的像素位置
#             black_mask = np.all(edge_img <= threshold, axis=-1)
            
#             # 在接近黑色像素位置，用 validation 图像的像素替换
#             edge_img[black_mask] = validation_img[black_mask]
            
#             # 保存合并后的图像
#             cv2.imwrite(output_path, edge_img)
#             print(f"合并完成：{output_path}")
