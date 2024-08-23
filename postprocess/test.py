import numpy as np
from PIL import Image

def check_pixels_on_line(image_array, start_point, end_point):
    x1, y1 = start_point
    x2, y2 = end_point
    line_pixels = []
    # 使用 Bresenham's line algorithm 获取连线上的所有像素
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        pixel = image_array[y1, x1]
        if (pixel == [224, 255, 192]).all() or (pixel == [0, 0, 0]).all():
            return False  # 如果找到不允许的像素，返回False
        
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return True  # 没有找到不允许的像素，返回True

def main():
    image_path = "/data1/JM/code/mask2former/postprocess/result/annotated_output_image.png"
    image = Image.open(image_path)
    image_array = np.array(image)

    start_point = (203, 26)
    end_point = (250, 78)

    if check_pixels_on_line(image_array, start_point, end_point):
        print("The line does not touch the forbidden pixels.")
    else:
        print("The line touches the forbidden pixels.")

if __name__ == "__main__":
    main()
