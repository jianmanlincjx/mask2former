# import os
# import shutil

# def process_images(input_dir, output_dir, total_images, copies):
#     # 获取所有图像文件，并排序
#     images = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    
#     # 创建输出目录（如果不存在）
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # 计算每份图像的起始索引
#     for i in range(copies):
#         for j, img in enumerate(images):
#             new_index = (i * total_images) + j  # 计算新的文件名索引
#             new_name = f"{str(new_index).zfill(7)}.png"  # 生成新的文件名
#             shutil.copy(os.path.join(input_dir, img), os.path.join(output_dir, new_name))

# # 定义输入输出路径
# data_images_path = '/data1/JM/code/mask2former/post_process_data/data'
# label_images_path = '/data1/JM/code/mask2former/post_process_data/label'
# output_data_path = '/data1/JM/code/mask2former/post_process_data/processed_data'
# output_label_path = '/data1/JM/code/mask2former/post_process_data/processed_label'

# # 设置参数
# total_images = 7193  # 总图像数量
# copies = 10  # 复制的份数

# # 处理图像
# process_images(data_images_path, output_data_path, total_images, copies)
# process_images(label_images_path, output_label_path, total_images, copies)

# print("图像处理完成！")


# import matplotlib.pyplot as plt

# # 定义颜色映射
# floorplan_fuse_map = {
#     # wall 白色
#     1: (255, 255, 255),   
#     # door 绿色, double-door 墨绿色, sliding-door 深红色
#     2: [(0, 255, 0), (0, 126, 0), (126, 0, 0)],  
#     # window 红色, balcony-window 棕色
#     3: [(255, 0, 0), (126, 77, 7)],  
#     # bay-window 粉红色
#     4: (255, 191, 186), 
#     # livingroom 浅蓝色, doorway 深蓝色
#     5: [(3, 152, 255), (23, 0, 255), (251, 96, 255)],  
#     # bedroom 橙色
#     6: (254, 153, 3), 
#     # kitchen 黄色    
#     7: (0, 255, 255),    
#     # bathroom 灰色 
#     8: (127, 127, 127), 
#     # library 肤色  
#     9: (215, 210, 210), 
#     # balcony 灰绿色     
#     10: (98, 126, 91),  
#     # other 黑色  
#     11: (0, 0, 0),         
# }

# # 创建一个图形和轴
# fig, ax = plt.subplots(figsize=(10, 6))

# # 绘制每种颜色并添加数字标签
# for idx, color in floorplan_fuse_map.items():
#     if isinstance(color, list):
#         # 只取第一个颜色
#         first_color = color[0]
#     else:
#         first_color = color

#     # 在每个柱子之间增加间距
#     ax.add_patch(plt.Rectangle((idx * 1.5 - 1, 0), 1, 1, color=[x / 255 for x in first_color]))
#     ax.text(idx * 1.5 - 0.5, 0.5, str(idx), ha='center', va='center', fontsize=10, color='black')

# # 设置轴的范围和标签
# ax.set_xlim(0, len(floorplan_fuse_map) * 1.5)
# ax.set_ylim(0, 1)
# ax.axis('off')  # 关闭坐标轴

# # 显示图形
# plt.title("Floorplan Color Mapping with Numbers")
# plt.savefig('color.png')

import os

# 定义需要创建的输出目录
output_dirs = [
    "/data1/JM/code/mask2former/post_process_data/predict_label_1",
    "/data1/JM/code/mask2former/post_process_data/predict_label_2",
    "/data1/JM/code/mask2former/post_process_data/predict_label_3",
    "/data1/JM/code/mask2former/post_process_data/predict_label_4",
    "/data1/JM/code/mask2former/post_process_data/predict_label_5",
    "/data1/JM/code/mask2former/post_process_data/predict_label_6",
    "/data1/JM/code/mask2former/post_process_data/predict_label_7",
    "/data1/JM/code/mask2former/post_process_data/predict_label_8",
    "/data1/JM/code/mask2former/post_process_data/predict_label_9",
    "/data1/JM/code/mask2former/post_process_data/predict_label_10"
]

# 创建这些目录
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)
    print(f"Directory created: {directory}")

print("All directories created successfully.")
