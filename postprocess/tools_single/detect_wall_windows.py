import csv
from collections import defaultdict
from itertools import combinations
import numpy as np
from PIL import Image
import cv2
import os

class WallWindowDetector:
    def __init__(self, image_path, csv_input_path, output_dir, forbidden_color=[224, 255, 192], tolerance=2):
        self.image_path = image_path
        self.csv_input_path = csv_input_path
        self.output_dir = output_dir
        self.forbidden_color = np.array(forbidden_color)
        self.tolerance = tolerance
        self.image = None
        self.image_array = None
        self.image_cv = None
        self.grouped_data = defaultdict(list)
        self.walls = []
        self.windows = []

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_image(self):
        self.image = Image.open(self.image_path)
        self.image_array = np.array(self.image)
        self.image_cv = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)

    def load_csv_data(self):
        data = []
        with open(self.csv_input_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row['Position'] = self.parse_position(row['Position'])
                data.append(row)

        for item in data:
            self.grouped_data[item['Type']].append((item['Position'], item['Color']))

    @staticmethod
    def parse_position(position_str):
        return tuple(map(int, position_str.strip("()").split(",")))

    def are_collinear(self, p1, p2):
        if abs(p1[0] - p2[0]) <= self.tolerance:
            x = p1[0]
            y_start, y_end = sorted([p1[1], p2[1]])
            for y in range(y_start, y_end + 1):
                if np.array_equal(self.image_array[y, x], self.forbidden_color):
                    return False
            return True
        elif abs(p1[1] - p2[1]) <= self.tolerance:
            y = p1[1]
            x_start, x_end = sorted([p1[0], p2[0]])
            for x in range(x_start, x_end + 1):
                if np.array_equal(self.image_array[y, x], self.forbidden_color):
                    return False
            return True
        return False
    
    def are_collinear_x_axis(self, p1, p2):
        """Check if two points are collinear along the x-axis (vertical)."""
        if abs(p1[0] - p2[0]) <= self.tolerance:
            x = p1[0]
            y_start, y_end = sorted([p1[1], p2[1]])
            for y in range(y_start, y_end + 1):
                if np.array_equal(self.image_array[y, x], self.forbidden_color):
                    return False
            return True
        return False

    def are_collinear_y_axis(self, p1, p2):
        """Check if two points are collinear along the y-axis (horizontal)."""
        if abs(p1[1] - p2[1]) <= self.tolerance:
            y = p1[1]
            x_start, x_end = sorted([p1[0], p2[0]])
            for x in range(x_start, x_end + 1):
                if np.array_equal(self.image_array[y, x], self.forbidden_color):
                    return False
            return True
        return False

    def process_walls_and_windows(self):
        if 'wall' in self.grouped_data:
            wall_positions = [pos for pos, color in self.grouped_data['wall']]
            wall_positions = sorted(wall_positions, key=lambda x: (x[0], x[1]))

            processed_points = {}  # Dictionary to track the axes on which each point has been processed

            while len(wall_positions) > 0:
                start_point = wall_positions.pop(0)
                if start_point not in processed_points:
                    processed_points[start_point] = {'x': False, 'y': False}
                
                x_axis_checked = processed_points[start_point]['x']
                y_axis_checked = processed_points[start_point]['y']

                # Check for walls along the x-axis (vertical)
                for i, end_point in enumerate(wall_positions):
                    if end_point not in processed_points:
                        processed_points[end_point] = {'x': False, 'y': False}

                    if not x_axis_checked and not processed_points[end_point]['x'] and self.are_collinear_x_axis(start_point, end_point):
                        self.walls.append((start_point, end_point, self.grouped_data['wall'][0][1]))
                        processed_points[start_point]['x'] = True  # Mark x-axis as processed for start_point
                        processed_points[end_point]['x'] = True  # Mark x-axis as processed for end_point
                        x_axis_checked = True  # Stop considering x-axis for this start_point in this loop
                        break  # Break out of the loop after finding a wall on x-axis

                # Check for walls along the y-axis (horizontal)
                for i, end_point in enumerate(wall_positions):
                    if end_point not in processed_points:
                        processed_points[end_point] = {'x': False, 'y': False}

                    if not y_axis_checked and not processed_points[end_point]['y'] and self.are_collinear_y_axis(start_point, end_point):
                        self.walls.append((start_point, end_point, self.grouped_data['wall'][0][1]))
                        processed_points[start_point]['y'] = True  # Mark y-axis as processed for start_point
                        processed_points[end_point]['y'] = True  # Mark y-axis as processed for end_point
                        y_axis_checked = True  # Stop considering y-axis for this start_point in this loop
                        break  # Break out of the loop after finding a wall on y-axis
                    
        if 'windows' in self.grouped_data:
            window_positions = [pos for pos, color in self.grouped_data['windows']]
            window_positions = sorted(window_positions, key=lambda x: (x[0], x[1]))
            
            while len(window_positions) > 1:
                start_point = window_positions.pop(0)
                
                for i, end_point in enumerate(window_positions):
                    if self.are_collinear(start_point, end_point):
                        self.windows.append((start_point, end_point, self.grouped_data['windows'][0][1]))
                        window_positions.pop(i)
                        break

    def save_results_to_csv(self, output_csv_path):
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['Type', 'Start_Position', 'End_Position', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, (pos1, pos2, color) in enumerate(self.walls, start=1):
                writer.writerow({'Type': 'wall', 'Start_Position': pos1, 'End_Position': pos2, 'Color': color})
            for i, (pos1, pos2, color) in enumerate(self.windows, start=1):
                writer.writerow({'Type': 'window', 'Start_Position': pos1, 'End_Position': pos2, 'Color': color})

    def draw_and_save(self, start_point, end_point, color, name_prefix):
        image_copy = self.image_cv.copy()
        cv2.line(image_copy, start_point, end_point, [255,0,255], thickness=5)
        filename = f"{name_prefix}_{start_point[0]}_{start_point[1]}_{end_point[0]}_{end_point[1]}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, image_copy)
        print(f"Saved: {filepath}")

    def visualize_and_save(self):
        for start_point, end_point, color in self.walls:
            self.draw_and_save(start_point, end_point, [255,0,255], f"wall")

        for start_point, end_point, color in self.windows:
            self.draw_and_save(start_point, end_point, [0,255,0], f"window")

    def run(self):
        self.load_image()
        self.load_csv_data()
        self.process_walls_and_windows()
        self.save_results_to_csv(os.path.join(self.output_dir, 'detected_walls_windows.csv'))
        self.visualize_and_save()


# Example usage
if __name__ == "__main__":
    detector = WallWindowDetector(
        image_path='/data1/JM/code/mask2former/postprocess/result/cropped_output_image.png',
        csv_input_path='/data1/JM/code/mask2former/postprocess/result/filtered_corners_info.csv',
        output_dir='/data1/JM/code/mask2former/postprocess/result/result',
        forbidden_color=[224, 255, 192],
        tolerance=2
    )
    detector.run()
