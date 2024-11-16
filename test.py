import os
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import binary_dilation

# 读取图片并去除Alpha通道
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')  # 转换为RGB格式
    return np.array(img)

# 处理类别组1（wall, door, window）
def calculate_class_accuracy_group1(segmentation_result, segmentation_label, fuse_map_group1, fuse_map_group2, threshold=0.90):
    class_accuracies = {}

    # 获取客厅和厨房的颜色定义
    kitchen_color = fuse_map_group2[11]  # kitchen
    livingroom_color = fuse_map_group2[9]  # livingroom

    # 获取标签中客厅和厨房的掩码
    kitchen_mask = np.all(segmentation_label == kitchen_color, axis=-1)
    livingroom_mask = np.all(segmentation_label == livingroom_color, axis=-1)

    # 对客厅和厨房的掩码进行膨胀操作
    dilation_structure = np.ones((18, 18), dtype=bool)  # 可根据需要调整膨胀结构的大小
    kitchen_dilated = binary_dilation(kitchen_mask, structure=dilation_structure)
    livingroom_dilated = binary_dilation(livingroom_mask, structure=dilation_structure)

    # 合并膨胀后的客厅和厨房区域
    combined_mask = kitchen_dilated | livingroom_dilated


    # 对于墙、门、窗类别，计算在膨胀区域内的分类准确率
    for class_id, color in fuse_map_group1.items():
        # 获取结果和标签中该类别的像素掩码
        result_mask = np.all(segmentation_result == color, axis=-1)
        label_mask = np.all(segmentation_label == color, axis=-1)

        # 仅在膨胀后的区域内进行计算
        result_mask = result_mask & combined_mask
        label_mask = label_mask & combined_mask

        # result_mask_uint8 = result_mask.astype(np.uint8) * 255
        # cv2.imwrite('result_mask_uint8.png', result_mask_uint8)


        # label_mask_uint8 = label_mask.astype(np.uint8) * 255
        # cv2.imwrite('label_mask_uint8.png', label_mask_uint8)

        # exit()
        # 计算该类别的总像素数和匹配的像素数
        total_pixels = np.sum(label_mask)
        correct_pixels = np.sum(result_mask & label_mask)

        if total_pixels > 0:
            # 计算分类准确率
            accuracy = correct_pixels / total_pixels
            # 如果准确率超过阈值，返回1，否则返回0
            class_accuracies[class_id] = 1 if accuracy >= threshold else 0
        else:
            # 如果该类别在当前区域中不存在，忽略它
            class_accuracies[class_id] = None

    return class_accuracies

# 处理类别组2（kitchen, livingroom）
def calculate_class_accuracy_group2(segmentation_result, segmentation_label, fuse_map, threshold=0.90):
    class_accuracies = {}
    for class_id, color in fuse_map.items():
        # 获取结果和标签中该类别的像素掩码
        result_mask = np.all(segmentation_result == color, axis=-1)
        label_mask = np.all(segmentation_label == color, axis=-1)

        # 计算该类别的总像素数和匹配的像素数
        total_pixels = np.sum(label_mask)
        correct_pixels = np.sum(result_mask & label_mask)

        if total_pixels > 0:
            # 计算分类准确率
            accuracy = correct_pixels / total_pixels
            # 如果准确率超过阈值，返回1，否则返回0
            class_accuracies[class_id] = 1 if accuracy >= threshold else 0
        else:
            class_accuracies[class_id] = None

    return class_accuracies

# 批量处理所有PNG图片并计算每个类别的分类准确性
def batch_process_images(result_dir, label_dir, fuse_map_group1, fuse_map_group2, threshold_1=0.90, threshold_2=0.90):
    # 用于累积每个类别的分类结果（0或1）
    category_sums_group1 = {class_id: 0 for class_id in fuse_map_group1}
    category_counts_group1 = {class_id: 0 for class_id in fuse_map_group1}

    category_sums_group2 = {class_id: 0 for class_id in fuse_map_group2}
    category_counts_group2 = {class_id: 0 for class_id in fuse_map_group2}

    for file_name in os.listdir(result_dir):

        result_image_path = os.path.join(result_dir, file_name)
        label_image_path = os.path.join(label_dir, file_name).replace('jpg', 'png')

        # 加载语义分割结果和标签
        segmentation_result = load_image(result_image_path)
        segmentation_label = load_image(label_image_path)

        # 计算当前图像中类别组1的分类结果（仅在客厅和厨房周围）
        class_accuracies_group1 = calculate_class_accuracy_group1(
            segmentation_result, segmentation_label, fuse_map_group1, fuse_map_group2, threshold_1
        )

        # 计算当前图像中类别组2的分类结果
        class_accuracies_group2 = calculate_class_accuracy_group2(
            segmentation_result, segmentation_label, fuse_map_group2, threshold_2
        )

        # 累积类别组1的分类结果
        for class_id, accuracy in class_accuracies_group1.items():
            if accuracy is not None:
                category_sums_group1[class_id] += accuracy
                category_counts_group1[class_id] += 1

        # 累积类别组2的分类结果
        for class_id, accuracy in class_accuracies_group2.items():
            if accuracy is not None:
                category_sums_group2[class_id] += accuracy
                category_counts_group2[class_id] += 1

    # 计算每个类别的平均分类结果
    average_accuracies_group1 = {}
    for class_id in fuse_map_group1:
        if category_counts_group1[class_id] > 0:
            average_accuracies_group1[class_id] = category_sums_group1[class_id] / category_counts_group1[class_id]
        else:
            average_accuracies_group1[class_id] = None

    average_accuracies_group2 = {}
    for class_id in fuse_map_group2:
        if category_counts_group2[class_id] > 0:
            average_accuracies_group2[class_id] = category_sums_group2[class_id] / category_counts_group2[class_id]
        else:
            average_accuracies_group2[class_id] = None

    return average_accuracies_group1, average_accuracies_group2

# 主函数
result_dir = '/data1/JM/code/mask2former/金牌测试/conver_images第一批-完成47组/result'
label_dir = '/data1/JM/code/mask2former/金牌测试/conver_images第一批-完成47组/seg_images'

# 定义类别映射
fuse_map_group1 = {
    1: (255, 255, 255),   # wall 白色
    2: (0, 255, 0),       # door 绿色
    5: (255, 0, 0),       # window 红色
}

fuse_map_group2 = {
    11: (0, 255, 255),    # kitchen 黄色
    9: (3, 152, 255),     # livingroom 浅蓝色
    10: (254, 153, 3),     # bedroom 橙色
    12: (127, 127, 127),   # bathroom 灰色
    13: (215, 210, 210),   # library 肤色
    14: (98, 126, 91),     # balcony 灰绿色        
    15: (251, 96, 255),    # diningroom 紫色
}

# 批量处理并计算每个类别的分类结果平均值
average_accuracies_group1, average_accuracies_group2 = batch_process_images(
    result_dir, label_dir, fuse_map_group1, fuse_map_group2, threshold_1=0.50, threshold_2=0.50
)

# 输出每个类别的平均分类结果
class_names = {1: 'wall', 2: 'door', 5: 'window', 9: 'livingroom', 10: 'bedroom', 11: 'kitchen', 12: 'bathroom', 13: 'library', 14: 'balcony', 15: 'diningroom'}

print("类别组1（wall, door, window，位于客厅和厨房周围）结果：")
for class_id, avg_accuracy in average_accuracies_group1.items():
    class_name = class_names[class_id]
    if avg_accuracy is not None:
        print(f"{class_name} (class {class_id}) 在图像中被正确分类的比例: {avg_accuracy:.4f}")
    else:
        print(f"{class_name} (class {class_id}) 在所有图像中都不存在。")

print("\n类别组2（kitchen, livingroom）结果：")
for class_id, avg_accuracy in average_accuracies_group2.items():
    class_name = class_names[class_id]
    if avg_accuracy is not None:
        print(f"{class_name} (class {class_id}) 在图像中被正确分类的比例: {avg_accuracy:.4f}")
    else:
        print(f"{class_name} (class {class_id}) 在所有图像中都不存在。")

# 彩底 按90%重叠算
# (mask2former) (base) wubowen@exai-166:/data1/JM/code/mask2former$ python test.py 
# 类别组1（wall, door, window，位于客厅和厨房周围）结果：
# wall (class 1) 在图像中被正确分类的比例: 0.8862
# door (class 2) 在图像中被正确分类的比例: 1.0000
# window (class 5) 在图像中被正确分类的比例: 0.9873

# 类别组2（kitchen, livingroom）结果：
# kitchen (class 11) 在图像中被正确分类的比例: 0.9128
# livingroom (class 9) 在图像中被正确分类的比例: 0.9699
# bedroom (class 10) 在图像中被正确分类的比例: 0.8870
# bathroom (class 12) 在图像中被正确分类的比例: 0.9116
# library (class 13) 在图像中被正确分类的比例: 0.9833
# balcony (class 14) 在图像中被正确分类的比例: 0.3858
# diningroom (class 15) 在图像中被正确分类的比例: 1.0000

######################################################################################
######################################################################################
######################################################################################
######################################################################################
# 白底 75%
# (mask2former) (base) wubowen@exai-166:/data1/JM/code/mask2former$ python test.py 
# 类别组1（wall, door, window，位于客厅和厨房周围）结果：
# wall (class 1) 在图像中被正确分类的比例: 0.4722
# door (class 2) 在图像中被正确分类的比例: 0.6253
# window (class 5) 在图像中被正确分类的比例: 0.5678

# 类别组2（kitchen, livingroom）结果：
# kitchen (class 11) 在图像中被正确分类的比例: 0.5635
# livingroom (class 9) 在图像中被正确分类的比例: 0.7924
# bedroom (class 10) 在图像中被正确分类的比例: 0.6733
# bathroom (class 12) 在图像中被正确分类的比例: 0.6965
# library (class 13) 在图像中被正确分类的比例: 0.0000
# balcony (class 14) 在图像中被正确分类的比例: 0.0000
# diningroom (class 15) 在图像中被正确分类的比例: 0.0000

######################################################################################
######################################################################################
######################################################################################
######################################################################################

# 黑底 75%
# (mask2former) (base) wubowen@exai-166:/data1/JM/code/mask2former$ python test.py 
# 类别组1（wall, door, window，位于客厅和厨房周围）结果：
# wall (class 1) 在图像中被正确分类的比例: 0.0000
# door (class 2) 在图像中被正确分类的比例: 0.0000
# window (class 5) 在图像中被正确分类的比例: 0.0000

# 类别组2（kitchen, livingroom）结果：
# kitchen (class 11) 在图像中被正确分类的比例: 0.1515
# livingroom (class 9) 在图像中被正确分类的比例: 0.7045
# bedroom (class 10) 在图像中被正确分类的比例: 0.3778
# bathroom (class 12) 在图像中被正确分类的比例: 0.4074
# library (class 13) 在图像中被正确分类的比例: 0.0000
# balcony (class 14) 在图像中被正确分类的比例: 0.0000
# diningroom (class 15) 在图像中被正确分类的比例: 0.0000

# 黑底 65%
# (mask2former) (base) wubowen@exai-166:/data1/JM/code/mask2former$ python test.py 
# 类别组1（wall, door, window，位于客厅和厨房周围）结果：
# wall (class 1) 在图像中被正确分类的比例: 0.0000
# door (class 2) 在图像中被正确分类的比例: 0.0000
# window (class 5) 在图像中被正确分类的比例: 0.0000

# 类别组2（kitchen, livingroom）结果：
# kitchen (class 11) 在图像中被正确分类的比例: 0.3939
# livingroom (class 9) 在图像中被正确分类的比例: 0.8864
# bedroom (class 10) 在图像中被正确分类的比例: 0.7111
# bathroom (class 12) 在图像中被正确分类的比例: 0.5185
# library (class 13) 在图像中被正确分类的比例: 0.0000
# balcony (class 14) 在图像中被正确分类的比例: 0.0000
# diningroom (class 15) 在图像中被正确分类的比例: 0.0000

# 黑底 50%
# (mask2former) (base) wubowen@exai-166:/data1/JM/code/mask2former$ python test.py 
# 类别组1（wall, door, window，位于客厅和厨房周围）结果：
# wall (class 1) 在图像中被正确分类的比例: 0.0455
# door (class 2) 在图像中被正确分类的比例: 0.0000
# window (class 5) 在图像中被正确分类的比例: 0.0000

# 类别组2（kitchen, livingroom）结果：
# kitchen (class 11) 在图像中被正确分类的比例: 0.5455
# livingroom (class 9) 在图像中被正确分类的比例: 0.9318
# bedroom (class 10) 在图像中被正确分类的比例: 0.8889
# bathroom (class 12) 在图像中被正确分类的比例: 0.6667
# library (class 13) 在图像中被正确分类的比例: 0.0000
# balcony (class 14) 在图像中被正确分类的比例: 0.0000
# diningroom (class 15) 在图像中被正确分类的比例: 0.0000