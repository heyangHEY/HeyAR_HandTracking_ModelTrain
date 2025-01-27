import argparse
import json
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

def generate_colors(num_colors, reverse=False):
    cmap = plt.get_cmap('rainbow')
    if reverse:
        colors = [cmap(1 - i / num_colors) for i in range(num_colors)]  # 反转颜色映射
    else:
        colors = [cmap(i / num_colors) for i in range(num_colors)]
    # 转换为0-255范围的BGR颜色用于OpenCV
    colors_bgr = [(int(c[2]*255), int(c[1]*255), int(c[0]*255)) for c in colors]
    return colors, colors_bgr

def depth_two_uint8_to_float(top_bits, bottom_bits):
    """将RGB编码的深度图转换为浮点数深度值"""
    depth_map = (top_bits * 256 + bottom_bits).astype('float32')
    depth_map /= float(65535)
    depth_map *= 5.0
    return depth_map

def parse_args():
    parser = argparse.ArgumentParser(description="COCO格式标注数据的可视化")
    parser.add_argument('--image_dir', type=str, required=True, help='图片目录')
    parser.add_argument('--ann_file', type=str, required=True, help='COCO标注文件')
    return parser.parse_args()

def main():
    args = parse_args()
    coco = COCO(args.ann_file)

    # 21个关键点富裕彩虹色，方便比对
    colors_rgb, colors_bgr = generate_colors(21, reverse=True)  # 根据需要调整reverse参数
    
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

    # 使用index_map重新映射colors_rgb
    mapped_colors_rgb = np.zeros_like(colors_rgb)
    for i in range(21):
        mapped_colors_rgb[i] = colors_rgb[index_map.get(i, i)]

    # 获取所有图像的ID
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)

    # 加载类别信息并提取骨架
    categories = coco.loadCats(coco.getCatIds())
    cat_skeletons = {cat['id']: cat['skeleton'] for cat in categories}

    for img in imgs:
        print(f"{img['file_name']}")

        # 解析文件路径
        image_path = os.path.join(args.image_dir, img['file_name'])
        depth_path = os.path.join(args.image_dir, img['file_name'].replace('color', 'depth'))
        mask_path = os.path.join(args.image_dir, img['file_name'].replace('color', 'mask'))

        # 加载图像、深度图和掩码图
        image = np.array(Image.open(image_path))
        depth = np.array(Image.open(depth_path))
        mask = np.array(Image.open(mask_path))

        # 处理深度信息
        depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])

        # 获取该图像的注解
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)

        # 相机内参
        focal = img['cam_param']['focal']
        princpt = img['cam_param']['princpt']
        camera_intrinsic_matrix = np.array([  # 相机内参矩阵
            [focal[0], 0, princpt[0]],
            [0, focal[1], princpt[1]],
            [0, 0, 1]
        ])

        # 可视化
        fig = plt.figure(figsize=(20, 15), num='coco annotations')
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')

        ax1.imshow(image)
        ax2.imshow(depth)
        ax3.imshow(mask)

        for ann in anns:
            # 绘制bbox
            bbox = ann['bbox']
            x, y, w, h = bbox
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax1.add_patch(rect)

            # 绘制关键点
            keypoints = np.array(ann['keypoints']).reshape(-1, 3)
            visible_keypoints = keypoints[keypoints[:, 2] > 0]
            # ax1.plot(visible_keypoints[:, 0], visible_keypoints[:, 1], 'ro')
            # 使用scatter绘制每个关键点，允许每个点有不同的颜色
            # ax1.scatter(visible_keypoints[:, 0], visible_keypoints[:, 1], color=mapped_colors_rgb[keypoints[:, 2] > 0], s=10)  # s是点的大小
            ax1.scatter(keypoints[:, 0], keypoints[:, 1], color=mapped_colors_rgb, s=10)  # s是点的大小

            # 将世界坐标投影到相机坐标系
            kp_coord_xyz = np.array(ann['joint_cam']).reshape(-1, 3)  # 关键点的x, y, z坐标，单位为米
            # kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
            # kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]
            # ax1.plot(kp_coord_uv_proj[:, 0], kp_coord_uv_proj[:, 1], 'gx')

            # 绘制关键点和骨架
            skeleton = cat_skeletons[ann['category_id']]
            for limb in skeleton:
                start_idx = index_map.get(limb[0]-1, limb[0]-1)  # 应用映射并调整索引
                end_idx = index_map.get(limb[1]-1, limb[1]-1)
                start_point = kp_coord_xyz[start_idx]
                end_point = kp_coord_xyz[end_idx]
                ax4.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 'r')

                # 标记关键点ID
                ax4.text(start_point[0], start_point[1], start_point[2], f'{start_idx}', color='blue')
                ax4.text(end_point[0], end_point[1], end_point[2], f'{end_idx}', color='blue')

            ax4.scatter(kp_coord_xyz[:, 0], kp_coord_xyz[:, 1], kp_coord_xyz[:, 2], c='b')

            ax4.view_init(azim=-90.0, elev=-90.0)
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            ax4.set_zlabel('z')

        plt.show()

if __name__ == "__main__":
    main()
