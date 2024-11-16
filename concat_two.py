import os
from PIL import Image

# 设置文件夹路径
image_folder = "/data1/JM/code/mask2former/金牌测试/input/链家_彩色"
label_folder = "/data1/JM/code/mask2former/客户彩底"
output_folder = "/data1/JM/code/mask2former/客户彩底_c"  # 输出拼接后的图像

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取所有图像文件名
image_files = sorted(os.listdir(image_folder))
label_files = sorted(os.listdir(label_folder))
print(len(image_files))
print(len(label_files))
# 确保两个文件夹里的文件数量一致
if len(image_files) != len(label_files):
    raise ValueError("两个文件夹中的文件数量不匹配！")

# 遍历每一对文件进行拼接
for img_file, label_file in zip(image_files, label_files):
    img_path = os.path.join(image_folder, img_file)
    label_path = os.path.join(label_folder, label_file)
    
    # 打开图片和标签
    img = Image.open(img_path)
    label = Image.open(label_path)
    
    # 确保图像和标签的高度相同（如果需要的话可以在这里做处理）
    if img.size[1] != label.size[1]:
        raise ValueError(f"图像 {img_file} 和标签 {label_file} 的高度不一致！")
    
    # 拼接图像（左右拼接）
    merged = Image.new('RGB', (img.width + label.width, img.height))
    merged.paste(img, (0, 0))  # 将图片粘贴到左边
    merged.paste(label, (img.width, 0))  # 将标签粘贴到右边
    
    # 保存拼接后的图像
    output_path = os.path.join(output_folder, f"merged_{img_file}")
    merged.save(output_path)

print("拼接完成！")
