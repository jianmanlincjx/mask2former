import torch
import numpy as np
import open3d as o3d
import pandas as pd

def generate_3d_model_from_csv(file_path, save_path, device="cuda", border_size=0.01, border_color=(1, 1, 1),
                               wall_height=1, window_base_height=0.3, window_height=0.7, scale_factors=(2, 1, 1)):
    def read_csv_to_data_format(file_path):
        df = pd.read_csv(file_path)
        data = []
        for index in range(len(df)):
            item_type = df.iloc[index]['Type']
            start_position = df.iloc[index]['Start_Position']
            end_position = df.iloc[index]['End_Position']
            color = df.iloc[index]['Color']
            start_position_formatted = f"({int(start_position.split(',')[0][1:])}, {int(start_position.split(',')[1][:-1])})"
            end_position_formatted = f"({int(end_position.split(',')[0][1:])}, {int(end_position.split(',')[1][:-1])})"
            data.append((item_type, start_position_formatted, end_position_formatted, color))
        return data

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

    data = read_csv_to_data_format(file_path)

    points = []
    for item in data:
        start_pos = parse_position(item[1])
        end_pos = parse_position(item[2])
        points.append(start_pos)
        points.append(end_pos)

    min_x = min([p[0] for p in points])
    min_y = min([p[1] for p in points])
    max_x = max([p[0] for p in points])
    max_y = max([p[1] for p in points])

    min_values = (min_x, min_y)
    max_values = (max_x, max_y)

    all_points = []
    for item in data:
        item_type, start_pos, end_pos, color = item
        start_pos = parse_position(start_pos)
        end_pos = parse_position(end_pos)
        start_pos = normalize_coordinates([start_pos], min_values, max_values)[0]
        end_pos = normalize_coordinates([end_pos], min_values, max_values)[0]

        if item_type == "wall":
            points_3d = [
                [start_pos[0], start_pos[1], 0],
                [end_pos[0], end_pos[1], 0],
                [end_pos[0], end_pos[1], wall_height],
                [start_pos[0], start_pos[1], wall_height],
            ]
        elif item_type == "window":
            points_3d = [
                [start_pos[0], start_pos[1], window_base_height],
                [end_pos[0], end_pos[1], window_base_height],
                [end_pos[0], end_pos[1], window_height],
                [start_pos[0], start_pos[1], window_height],
            ]
        all_points.append(points_3d)

    scale = torch.tensor(scale_factors, dtype=torch.float32, device=device).unsqueeze(0)

    def add_surface(points, face_idx, color):
        eps = 1e-6  # 定义一个非常小的值，防止浮点数精度误差
        vertices = torch.tensor(points, dtype=torch.float32, device=device) * scale
        n = torch.cross(vertices[1] - vertices[0], vertices[2] - vertices[1])
        n /= torch.norm(n)
        u1 = vertices[1] - vertices[0]
        u2 = vertices[2] - vertices[1]
        u1 /= torch.norm(u1)
        u2 /= torch.norm(u2)
        sign = (float((n * vertices[0]).sum() < 0) - 0.5) * 2
        inner_vertices = torch.stack(
            [
                vertices[0] + border_size * (u1 + u2) + sign * eps * n,
                vertices[1] + border_size * (u2 - u1) + sign * eps * n,
                vertices[2] - border_size * (u1 + u2) + sign * eps * n,
                vertices[3] + border_size * (u1 - u2) + sign * eps * n,
            ]
        )
        vertices = torch.cat([vertices, inner_vertices], dim=0)
        faces = (
            torch.tensor(
                [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
                dtype=torch.int64,
                device=device,
            )
            + face_idx
        )
        face_idx += 8
        colors = torch.tensor(
            [border_color] * 4 + [color] * 4, dtype=torch.float32, device=device
        )
        return vertices, faces, colors, face_idx

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

    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    colors_np = colors.cpu().numpy()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
    mesh.triangles = o3d.utility.Vector3iVector(faces_np)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors_np)

    o3d.io.write_triangle_mesh(save_path, mesh)

# 使用接口生成3D模型
generate_3d_model_from_csv(
    file_path='/data1/JM/code/mask2former/postprocess/result/result/detected_walls_windows.csv',
    save_path='/data1/JM/code/mask2former/postprocess/result/room_models.ply',
    device="cuda",
    border_size=0.01,
    border_color=(1, 1, 1),
    wall_height=1,
    window_base_height=0.3,
    window_height=0.7,
    scale_factors=(2, 1, 1)
)
