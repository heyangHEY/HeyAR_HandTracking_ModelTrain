#-*-coding:utf-8-*-
import cv2
import os
import numpy as np
import argparse

def load_images_and_labels(images_path, labels_path):
    images = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.bmp'))])
    labels = [os.path.join(labels_path, os.path.splitext(os.path.basename(f))[0] + '.txt') for f in images]
    return images, labels

def display_image(img, labels, label_map, w, h):
    for label in labels:
        class_id, cx, cy, bw, bh = map(float, label)
        class_id = int(class_id)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 30, 30), 2)
        cv2.putText(img, label_map[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 55), 2)

def main(args):
    if args.label_mode == 'gesture':
        label_map = [
            "grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign", \
            "hand_heart", "hand_heart2", "little_finger", "middle_finger", "take_picture", \
            "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", \
            "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", \
            "two_up", "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture"]
    elif args.label_mode == 'hand':
        label_map = ["hand"]
    img_files, label_files = load_images_and_labels(args.images_path, args.labels_path)
    index = 0
    
    while True:
        img_path = img_files[index]
        label_path = label_files[index]
        img = cv2.imread(img_path)
        if img is None:
            continue
        w, h = img.shape[1], img.shape[0]
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                labels = [line.split() for line in file.read().splitlines()]
        display_image(img, labels, label_map, w, h)
        cv2.imshow('image', img)
        print(img_path)
        print(f'w: {w}, h: {h}')
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 83:  # Right key
            index = min(index + 1, len(img_files) - 1)
        elif key == 81:  # Left key
            index = max(index - 1, 0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Annotations Viewer")
    parser.add_argument("--images_path", type=str, help="Path to the directory containing images")
    parser.add_argument("--labels_path", type=str, help="Path to the directory containing label files")
    parser.add_argument("--label_mode", type=str, help="gesture or hand")
    args = parser.parse_args()
    main(args)

