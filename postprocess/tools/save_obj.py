import torch
import numpy as np
import open3d as o3d
import torch


# 使用通用读取方法
file_path = '/data1/JM/code/mask2former/postprocess/result/result/detected_walls_windows.csv'  # 将此路径替换为你的CSV文件路径

save_path = "/data1/JM/code/mask2former/postprocess/result/room_models.ply"
device = "cuda"
border_size = 0.01
border_color = (1, 1, 1)
eps = 0.001
wall_height = 1  # Height of the walls after normalization
window_base_height = 0.3  # Base height of the windows after normalization
window_height = 0.7  # Height of the windows after normalization

import pandas as pd

def read_csv_to_data_format(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 初始化目标列表
    data = []
    
    # 遍历DataFrame的每一行
    for index in range(len(df)):
        item_type = df.iloc[index]['Type']
        start_position = df.iloc[index]['Start_Position']
        end_position = df.iloc[index]['End_Position']
        color = df.iloc[index]['Color']
        
        # 格式化位置为指定的格式
        start_position_formatted = f"({int(start_position.split(',')[0][1:])}, {int(start_position.split(',')[1][:-1])})"
        end_position_formatted = f"({int(end_position.split(',')[0][1:])}, {int(end_position.split(',')[1][:-1])})"
        
        # 添加到data列表
        data.append((item_type, start_position_formatted, end_position_formatted, color))
    
    return data


data = read_csv_to_data_format(file_path)

def parse_position(position):
    return [int(coord) for coord in position.strip("()").split(",")]

def normalize_coordinates(points, min_values, max_values):
    return [
        [
            (p[0] - min_values[0]) / (max_values[0] - min_values[0]),
            (p[1] - min_values[1]) / (max_values[1] - min_values[1]),
        ]
        for p in points
    ]

# Extract all points and determine the range of coordinates
points = []
for item in data:
    start_pos = parse_position(item[1])
    end_pos = parse_position(item[2])
    points.append(start_pos)
    points.append(end_pos)

# Find the minimum and maximum x and y values
min_x = min([p[0] for p in points])
min_y = min([p[1] for p in points])
max_x = max([p[0] for p in points])
max_y = max([p[1] for p in points])

min_values = (min_x, min_y)
max_values = (max_x, max_y)

# Normalize all points to the range [0, 1]
all_points = []

for item in data:
    item_type, start_pos, end_pos, color = item
    start_pos = parse_position(start_pos)
    end_pos = parse_position(end_pos)

    # Normalize coordinates
    start_pos = normalize_coordinates([start_pos], min_values, max_values)[0]
    end_pos = normalize_coordinates([end_pos], min_values, max_values)[0]

    if item_type == "wall":
        # Convert to 3D by adding height for walls
        points_3d = [
            [start_pos[0], start_pos[1], 0],
            [end_pos[0], end_pos[1], 0],
            [end_pos[0], end_pos[1], wall_height],
            [start_pos[0], start_pos[1], wall_height],
        ]
    elif item_type == "window":
        # Convert to 3D by placing windows at a specific height
        points_3d = [
            [start_pos[0], start_pos[1], window_base_height],
            [end_pos[0], end_pos[1], window_base_height],
            [end_pos[0], end_pos[1], window_height],
            [start_pos[0], start_pos[1], window_height],
        ]

    all_points.append(points_3d)  # Changed to directly append the list


device = "cuda"
border_size = 0.01
border_color = (1.0, 1.0, 1.0)
eps = 0.001

L, H, W = 2, 1, 1  # x, y, z
scale = torch.tensor([L, H, W], dtype=torch.float32, device=device).unsqueeze(0)

def add_surface(points, face_idx, color):
    vertices = torch.tensor(points, dtype=torch.float32, device=device) * scale

    # 平面法向量
    n = torch.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
    n /= torch.norm(n)

    # 边向量
    u1 = vertices[1] - vertices[0]
    u2 = vertices[2] - vertices[1]
    u1 /= torch.norm(u1)
    u2 /= torch.norm(u2)

    # 判断原点在平面法向量正方向还是负方向
    sign = (float((n * vertices[0]).sum() < 0) - 0.5) * 2

    # 子平面
    inner_vertices = torch.stack(
        [
            vertices[0] + border_size * (u1 + u2) + sign * eps * n,
            vertices[1] + border_size * (u2 - u1) + sign * eps * n,
            vertices[2] - border_size * (u1 + u2) + sign * eps * n,
            vertices[3] + border_size * (u1 - u2) + sign * eps * n,
        ]
    )

    # 合并顶点
    vertices = torch.cat([vertices, inner_vertices], dim=0)

    # 构建面三角
    faces = (
        torch.tensor(
            [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
            dtype=torch.int64,
            device=device,
        )
        + face_idx
    )
    face_idx += 8

    # 面的颜色
    colors = torch.tensor(
        [border_color] * 4 + [color] * 4, dtype=torch.float32, device=device
    )

    return vertices, faces, colors, face_idx



# 颜色定义
wall_color = (0.545, 0.271, 0.075)  # 褐色（Brown）
window_color = (0.0, 1.0, 0.0)  # 绿色（Green）

vertices = []
faces = []
colors = []
face_idx = 0

for points in all_points:
    _vertices, _faces, _colors, face_idx = add_surface(
        points, face_idx, color=(0, 0, 0)
    )
    vertices.append(_vertices)
    faces.append(_faces)
    colors.append(_colors)

vertices = torch.cat(vertices, dim=0)
faces = torch.cat(faces, dim=0)
colors = torch.cat(colors, dim=0)

# 转换为 NumPy 数组
vertices_np = vertices.cpu().numpy()
faces_np = faces.cpu().numpy()
colors_np = colors.cpu().numpy()

# 创建 Open3D 三角形网格
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
mesh.triangles = o3d.utility.Vector3iVector(faces_np)
mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

# 保存为 .ply 文件
o3d.io.write_triangle_mesh(save_path, mesh)
