import os
import pickle
import argparse
import json
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description="将RAW标注转换为COCO格式")
parser.add_argument('--in_path', type=str, required=True, help='RAW标注文件的路径')
parser.add_argument('--out_file', type=str, required=True, help='COCO标注文件')
args = parser.parse_args()

# 设置数据集目录
set_name = args.in_path # 'evaluation'或'training'

# 加载注释数据
with open(os.path.join(set_name, 'anno_{}.pickle'.format(set_name)), 'rb') as file:
    annotations = pickle.load(file)

# 定义COCO格式的基础结构
coco_format = {
    "info": {
        "description": "RHD",
        "version": "1.1",
        "year": "2025",
        "date_created": "2025/01/24"
    },
    "licenses": "",
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "left",
            "supercategory": "hand",
            "keypoints": ["wrist", "thumb1", "thumb2", "thumb3", "thumb4",
                          "forefinger1", "forefinger2", "forefinger3", "forefinger4",
                          "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4",
                          "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4",
                          "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"],
            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9],
                         [1, 10], [10, 11], [11, 12], [12, 13], [1, 14], [14, 15], [15, 16],
                         [16, 17], [1, 18], [18, 19], [19, 20], [20, 21]]
        },
        {
            "id": 2,
            "name": "right",
            "supercategory": "hand",
            "keypoints": ["wrist", "thumb1", "thumb2", "thumb3", "thumb4",
                          "forefinger1", "forefinger2", "forefinger3", "forefinger4",
                          "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4",
                          "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4",
                          "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"],
            "skeleton": [[1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9],
                         [1, 10], [10, 11], [11, 12], [12, 13], [1, 14], [14, 15], [15, 16],
                         [16, 17], [1, 18], [18, 19], [19, 20], [20, 21]]
        }
    ]
}

def remap_keypoints(keypoints):
    index_map = {
            0: 0,
            1: 4, 4: 1,
            2: 3, 3: 2,
            5: 8, 8: 5,
            6: 7, 7: 6,
            9: 12, 12: 9,
            10: 11, 11: 10,
            13: 16, 16: 13,
            14: 15, 15: 14,
            17: 20, 20: 17,
            18: 19, 19: 18
        }
    new_keypoints = np.array([keypoints[index_map[i]] for i in range(21)])
    return new_keypoints

# 用于跟踪注释ID
annotation_id = 0

for sample_id, anno in annotations.items():
    # 获取图像信息
    image_path = os.path.join(set_name, 'color', '{:05d}.png'.format(sample_id))
    image = Image.open(image_path)
    width, height = image.size # TODO check一下对不对

    camera_intrinsic_matrix = anno['K']  # 相机内参矩阵
    cam_param = {
        "focal": [float(camera_intrinsic_matrix[0, 0]), float(camera_intrinsic_matrix[1, 1])],
        "princpt": [float(camera_intrinsic_matrix[0, 2]), float(camera_intrinsic_matrix[1, 2])]
    }


    # 添加图像到COCO格式的数据
    coco_format["images"].append({
        "file_name": image_path,
        "height": height,
        "width": width,
        "id": sample_id,
        "cam_param": cam_param
    })

    # 处理左手和右手的关键点
    for hand_index, hand_name in enumerate(["left", "right"]):
        category_id = 1 if hand_name == "left" else 2
        keypoints = anno['uv_vis'][21*hand_index:21*(hand_index+1)]
        visibility = keypoints[:, 2]
        keypoints = remap_keypoints(keypoints)

        # 计算关键点的包围盒
        # 如果21个点都不可视，则不添加该注释
        # 计算bbox时，考虑所有21个点，不管是否可视
        if np.any(visibility > 0):  # 至少有一个点是可见的
            min_x, min_y = np.min(keypoints[:, :2], axis=0)
            max_x, max_y = np.max(keypoints[:, :2], axis=0)
            # TODO 严格来说也许应该在mask中找边界。不知openmmlab是如何处理的
            # 上下各往外5个像素
            min_x = max(min_x - 5, 0)
            min_y = max(min_y - 5, 0)
            max_x = min(max_x + 5, width)
            max_y = min(max_y + 5, height)
            bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
        else:
            continue  # 如果没有可见点，跳过这个注释
        
        keypoints = [[float(kpt[0]), float(kpt[1]), int(kpt[2])] for kpt in keypoints]
        joint_cam = anno['xyz'][21*hand_index:21*(hand_index+1)]
        joint_cam = remap_keypoints(joint_cam)
        joint_cam = [[float(item*1000) for item in xyz] for xyz in joint_cam] # mm单位转m

        # 添加注释到COCO格式的数据
        coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": sample_id,
            "category_id": category_id,
            "iscrowd": 0,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "segmentation": [],
            "keypoints": keypoints,
            "joint_cam": joint_cam
        })
        annotation_id += 1

if not os.path.exists(os.path.dirname(args.out_file)):
    os.makedirs(os.path.dirname(args.out_file))

# 保存COCO格式的数据到JSON文件
with open(args.out_file, 'w') as f:
    json.dump(coco_format, f, indent=4)

print("Conversion complete. JSON saved to '{}'.".format(args.out_file))