记录一下尝试过的数据集，和处理脚本。

数据集：
1. hand_keypoint_dataset_26k
   1. 26k个样本，图片大小统一为224*224；
   2. 带hand分类标签（不区分左右手）、 21个hand关键点2d坐标标签；
   3. 提供了yolo format和coco format两种标注格式；
2. RHD(Rendered Hand Pose)
   1. 44k个样本，图片大小统一为320*320；
   2. 带hand分类标签（可区分左右手）、 21个hand关键点2d坐标标签 和 3d坐标标签；
   3. 提供了原始格式和coco格式的标注；