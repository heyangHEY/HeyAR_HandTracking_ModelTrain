import argparse
import os
import shutil

def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            parts = line.strip().split()
            parts[0] = '0'  # 将class_id设置为0
            f.write(' '.join(parts) + '\n')

def main(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入目录下的所有文件和文件夹
    for root, dirs, files in os.walk(input_dir):
        # 复制目录结构到输出目录
        for dir in dirs:
            new_dir = os.path.join(output_dir, os.path.relpath(os.path.join(root, dir), input_dir))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
        
        # 处理所有.txt文件
        for file in files:
            if file.endswith('.txt'):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, input_dir)
                output_file = os.path.join(output_dir, relative_path)
                process_file(input_file, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert class_id in YOLO formatted label files to 0.')
    parser.add_argument('--input_dir', type=str, help='Directory path containing label files.')
    parser.add_argument('--output_dir', type=str, help='Output directory path where modified files will be stored.')
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)
