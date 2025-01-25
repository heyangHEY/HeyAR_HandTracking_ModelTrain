from __future__ import print_function, unicode_literals
import argparse
import json
import os
import shutil
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description="将COCO标注转换为YOLO格式")
    parser.add_argument('--image_path', type=str, required=True, help='图片存储路径')
    parser.add_argument('--ann_file', type=str, required=True, help='COCO标注文件')
    parser.add_argument('--out', type=str, required=True, help='YOLO标注的输出路径')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test'], help='数据集模式（train/val/test）')
    return parser.parse_args()

def adjust_keypoints(keypoints, index_map):
    num_keypoints = 21
    adjusted_keypoints = []
    for i in range(num_keypoints):
        mapped_index = index_map.get(i, i)
        adjusted_keypoints.append(keypoints[mapped_index])
    return adjusted_keypoints

def convert_coco_to_yolo(coco, img_id, img_file_name, output_dir, index_map):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_details = coco.loadImgs(img_id)[0]
    width = img_details['width']
    height = img_details['height']

    yolo_data = []
    for ann in anns:
        cls_id = ann['category_id'] - 1 # id 从0开始计数
        bbox = ann['bbox']
        cx = (bbox[0] + bbox[2] / 2) / width
        cy = (bbox[1] + bbox[3] / 2) / height
        w = bbox[2] / width
        h = bbox[3] / height

        # 确保坐标和尺寸在0到1之间
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        keypoints = adjust_keypoints(ann['keypoints'], index_map)
        for i in range(21):
            keypoints[i][0] = keypoints[i][0] / width
            keypoints[i][1] = keypoints[i][1] / height
            if keypoints[i][0] < 0 or keypoints[i][0] > 1 or keypoints[i][1] < 0 or keypoints[i][1] > 1:
                keypoints[i][0] = 0 # 必须调整到 [0, 1] 范围内，否则yolo会将这个样本丢弃
                keypoints[i][1] = 0
                keypoints[i][2] = 0 # 代表不可见

        keypoints_str = ' '.join(str(item) for kpt in keypoints for item in kpt) # Flatten the list and convert to string

        yolo_format = f"{cls_id} {cx} {cy} {w} {h} {keypoints_str}"
        yolo_data.append(yolo_format)

    # Save to file
    file_name = os.path.splitext(os.path.basename(img_file_name))[0] + '.txt'
    with open(os.path.join(output_dir, file_name), 'w') as file:
        file.write("\n".join(yolo_data))

def main():
    args = parse_args()
    coco = COCO(args.ann_file)

    if not os.path.exists(os.path.join(args.out, 'images', args.mode)):
        os.makedirs(os.path.join(args.out, 'images', args.mode))
    if not os.path.exists(os.path.join(args.out, 'labels', args.mode)):
        os.makedirs(os.path.join(args.out, 'labels', args.mode))

    # 索引映射关系
    # ! 我已在raw转coco时将关键点索引做了转换，所以此处不用再做了
    index_map = {}
    # index_map = {
    #     1: 4, 4: 1,
    #     2: 3, 3: 2,
    #     5: 8, 8: 5,
    #     6: 7, 7: 6,
    #     9: 12, 12: 9,
    #     10: 11, 11: 10,
    #     13: 16, 16: 13,
    #     14: 15, 15: 14,
    #     17: 20, 20: 17,
    #     18: 19, 19: 18
    # }

    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)

    for img in imgs:
        img_file_name = os.path.basename(img['file_name'])
        img_path = os.path.join(args.image_path, img['file_name'])
        output_img_path = os.path.join(args.out, 'images', args.mode, img_file_name)
        shutil.copy(img_path, output_img_path)  # Copy the image file to the output directory

        convert_coco_to_yolo(coco, img['id'], img_file_name, os.path.join(args.out, 'labels', args.mode), index_map)

if __name__ == "__main__":
    main()
