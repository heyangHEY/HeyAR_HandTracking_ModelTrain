import cv2
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def generate_colors(num_colors):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    # 转换为0-255范围的BGR颜色用于OpenCV
    colors_bgr = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    return colors, colors_bgr

def parse_annotations(ann_path):
    annotations = []
    with open(ann_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, width, height = map(float, data[1:5])
        keypoints = []
        for i in range(5, len(data), 3):
            u = float(data[i])
            v = float(data[i+1])
            visible = int(data[i+2])
            keypoints.append((u, v, visible))
        annotations.append((class_id, x_center, y_center, width, height, keypoints))
    
    return annotations

def visualize(image_path, ann_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    annotations = parse_annotations(ann_path)
    
    for annotation in annotations:
        class_id, x_center, y_center, bbox_width, bbox_height, keypoints = annotation
        
        # Convert normalized coords to absolute coords
        x1 = int((x_center - bbox_width / 2) * width)
        y1 = int((y_center - bbox_height / 2) * height)
        x2 = int((x_center + bbox_width / 2) * width)
        y2 = int((y_center + bbox_height / 2) * height)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        for i, (u, v, visible) in enumerate(keypoints):
            u_int = int(u * width)  # Assume u, v are normalized
            v_int = int(v * height)
            cv2.circle(image, (u_int, v_int), 2, colors_bgr[i], -1)
            text_position = (u_int + 5, v_int)
            # cv2.putText(image, str(i), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors_bgr[i], 1)
    
    plt.figure(figsize=(2 * width / 100, 2 * height / 100), num='yolo annotations')  # DPI is roughly 100 by default
    plt.imshow(image)
    plt.show()

def main(root_dir):
    # 假设有n个关键点
    n = 21
    global colors_bgr
    colors_rgb, colors_bgr = generate_colors(n)

    for subdir in ['train', 'val', 'test']:
        img_dir = os.path.join(root_dir, 'images', subdir)
        ann_dir = os.path.join(root_dir, 'labels', subdir)
        
        if os.path.exists(img_dir) and os.path.exists(ann_dir):
            # Sort the files to ensure they are processed in order
            image_files = sorted(os.listdir(img_dir))
            for filename in image_files:
                image_path = os.path.join(img_dir, filename)
                ann_path = os.path.join(ann_dir, os.path.splitext(filename)[0] + '.txt')
                print(ann_path)
                if os.path.exists(ann_path):
                    visualize(image_path, ann_path)
        else:
            print(f"Directory {subdir} does not exist. Skipping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize YOLO dataset with bounding boxes and keypoints.')
    parser.add_argument('--root_dir', type=str, help='Root directory of the dataset')
    args = parser.parse_args()
    
    main(args.root_dir)
