# import os
# import shutil

# # 定义源文件夹和目标文件夹路径
# image_dirs = [
# '/data1/JM/code/mask2former/temp_black_w_color/20240904户型图200组-已校验',
# '/data1/JM/code/mask2former/temp_black_w_color/20240909户型图384组-已校验',
# '/data1/JM/code/mask2former/temp_black_w_color/20240912户型图391组-已校验',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批1阶段-完成67组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批2阶段-完成80组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批3阶段-完成79组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批4阶段-完成43组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批5阶段-完成65组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批6阶段-完成159组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批7阶段-完成73组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批1阶段-完成140组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批2阶段-完成80组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批3阶段-完成105组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批4阶段-完成105组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批5阶段-完成132组',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批6阶段-完成52组'
# ]

# label_dirs = [
# '/data1/JM/code/mask2former/temp_black_w_color/20240904户型图200组-已校验/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/20240909户型图384组-已校验/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/20240912户型图391组-已校验/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批1阶段-完成67组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批2阶段-完成80组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批3阶段-完成79组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批4阶段-完成43组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批5阶段-完成65组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批6阶段-完成159组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第四批7阶段-完成73组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批1阶段-完成140组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批2阶段-完成80组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批3阶段-完成105组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批4阶段-完成105组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批5阶段-完成132组/seg_images',
# '/data1/JM/code/mask2former/temp_black_w_color/户型图第五批6阶段-完成52组/seg_images'
# ]

# target_image_dir = '/data1/JM/code/mask2former/temp_black_w_color/all_image'
# target_label_dir = '/data1/JM/code/mask2former/temp_black_w_color/all_label'

# # 创建目标文件夹
# os.makedirs(target_image_dir, exist_ok=True)
# os.makedirs(target_label_dir, exist_ok=True)

# # 初始化起始编号
# start_number = 0

# # 定义允许的文件扩展名
# allowed_extensions = ['.png', '.jpg']

# # 遍历每个图像和标签文件夹，合并并检查配对
# for image_dir, label_dir in zip(image_dirs, label_dirs):
#     image_files = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and os.path.splitext(f)[1].lower() in allowed_extensions])
#     label_files = sorted([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and os.path.splitext(f)[1].lower() in allowed_extensions])
    
#     # 去掉扩展名，只使用文件名进行配对
#     image_file_names = {os.path.splitext(f)[0]: f for f in image_files}
#     label_file_names = {os.path.splitext(f)[0]: f for f in label_files}
    
#     # 遍历图像文件并检查是否有对应的标签文件
#     for image_name, image_file in image_file_names.items():
#         if image_name not in label_file_names:
#             print(f"标签文件缺失: {image_file}")
#             continue
        
#         # 获取原图像和标签文件路径
#         image_path = os.path.join(image_dir, image_file)
#         label_file = label_file_names[image_name]
#         label_path = os.path.join(label_dir, label_file)
        
#         # 生成新的文件名（统一为 .png 格式）
#         new_image_name = f'{start_number:06}.png'
#         new_label_name = new_image_name
        
#         # 移动图像和标签文件到目标目录
#         shutil.copy(image_path, os.path.join(target_image_dir, new_image_name))
#         shutil.copy(label_path, os.path.join(target_label_dir, new_label_name))
        
#         print(f'图像 {image_file} 和 标签 {label_file} 已重命名为 {new_image_name}')
        
#         # 更新编号
#         start_number += 1



import os
from PIL import Image

# 定义图像和标签目录
image_dir = '/data1/JM/code/mask2former/temp_black_w_color/all_image'
label_dir = '/data1/JM/code/mask2former/temp_black_w_color/all_label'

# 获取图像和标签文件列表
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg'))]
label_files = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f)) and f.endswith(('.png', '.jpg'))]

# 去掉扩展名，只保留文件名的主部分
image_files_base = {os.path.splitext(f)[0]: f for f in image_files}
label_files_base = {os.path.splitext(f)[0]: f for f in label_files}

# 只保留配对文件
paired_files = set(image_files_base.keys()).intersection(label_files_base.keys())

i = 0
deleted_count = 0
# 遍历配对文件，检查尺寸是否一致
print(len(image_files))
print(len(paired_files))
for base_name in paired_files:
    image_path = os.path.join(image_dir, image_files_base[base_name])
    label_path = os.path.join(label_dir, label_files_base[base_name])
    
    # 打开图像和标签文件
    image = Image.open(image_path)
    label = Image.open(label_path)
    
    # 获取图像和标签的尺寸
    image_shape = image.size  # 返回 (width, height)
    label_shape = label.size  # 返回 (width, height)
    
    # 比较尺寸
    if image_shape != label_shape:
        print(f"文件尺寸不一致: {base_name} - 图像尺寸: {image_shape}, 标签尺寸: {label_shape}")
        # 删除图像和标签文件
        os.remove(image_path)
        os.remove(label_path)
        print(f"已删除: {image_path} 和 {label_path}")
        deleted_count += 1
    i += 1

print(f"总共处理了 {i} 个配对文件")
print(f"总共删除了 {deleted_count} 个不匹配的文件对")
