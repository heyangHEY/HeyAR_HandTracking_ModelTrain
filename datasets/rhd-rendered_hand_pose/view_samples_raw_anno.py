import argparse
import pickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="RAW格式标注数据的可视化")
parser.add_argument('--mode', type=str, required=True, help='evaluation 或 training')
args = parser.parse_args()

# 选择数据集：训练集或评估集
set = args.mode # 'evaluation'或'training'

# 辅助函数：将RGB编码的深度图转换为浮点数深度值
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 256 + bottom_bits).astype('float32')
    depth_map /= float(65535)
    depth_map *= 5.0
    return depth_map

# 加载该数据集的注释
with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# 打印第一个样本的anno的键
sample_id, anno = next(iter(anno_all.items()))
# print("Keys in anno dictionary:", anno.keys()) # dict_keys(['K', 'xyz', 'uv_vis'])

# 遍历数据集中的样本
for sample_id, anno in anno_all.items():
    # 加载数据
    image = np.array(Image.open(os.path.join(set, 'color', '%.5d.png' % sample_id)))
    mask = np.array(Image.open(os.path.join(set, 'mask', '%.5d.png' % sample_id)))
    depth = np.array(Image.open(os.path.join(set, 'depth', '%.5d.png' % sample_id)))

    print(f'{sample_id}.png')

    # 处理RGB编码的深度为浮点数：红色通道为高位，绿色通道为低位
    depth = depth_two_uint8_to_float(depth[:, :, 0], depth[:, :, 1])  # 深度，单位为米

    # 从注释字典中获取信息
    # uv_vis、xyz 的长度都是42，其中前21个是left hand，后21个是right hand
    kp_coord_uv = anno['uv_vis'][:, :2]  # 手部关键点的u, v坐标
    kp_visible = (anno['uv_vis'][:, 2] == 1)  # 关键点是否可见
    kp_coord_xyz = anno['xyz']  # 关键点的x, y, z坐标，单位为米
    camera_intrinsic_matrix = anno['K']  # 相机内参矩阵

    # 将世界坐标投影到相机坐标系
    kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
    kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

    # 可视化数据
    fig = plt.figure(figsize=(20, 15), num='raw annotations')
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.imshow(image)
    ax1.plot(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], 'ro')
    ax1.plot(kp_coord_uv_proj[kp_visible, 0], kp_coord_uv_proj[kp_visible, 1], 'gx')
    ax2.imshow(depth)
    ax3.imshow(mask)
    ax4.scatter(kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
    ax4.view_init(azim=-90.0, elev=-90.0)  # 调整3D视图，使其与相机视角一致
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')

    plt.show()
