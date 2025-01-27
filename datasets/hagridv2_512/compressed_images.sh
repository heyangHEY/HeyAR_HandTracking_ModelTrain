#!/bin/bash

# 脚本用法: ./script.sh [compress|decompress]

operation=$1
base_dir=$(pwd)  # 假设脚本在项目根目录执行

case $operation in
  compress)
    echo "开始压缩操作..."

    # 压缩 images 子文件夹
    mkdir -p "${base_dir}/images_compressed"
    for folder in test train val; do
      mkdir -p "${base_dir}/images_compressed/${folder}"
      for subfolder in "${base_dir}/images/${folder}"/*; do
        if [ -d "$subfolder" ]; then
          base=$(basename "$subfolder")
          tar -czf "${base_dir}/images_compressed/${folder}/${base}.tar.gz" -C "${base_dir}/images/${folder}" "$base"
        fi
      done
    done

    # 压缩 labels 文件夹
    tar -czf "${base_dir}/labels.tar.gz" -C "${base_dir}" labels

    echo "压缩完成。"
    ;;

  decompress)
    echo "开始解压缩操作..."

    # 解压 images
    for folder in test train val; do
      for file in "${base_dir}/images_compressed/${folder}"/*.tar.gz; do
        tar -xzf "$file" -C "${base_dir}/images/${folder}"
      done
    done

    # 解压 labels
    tar -xzf "${base_dir}/labels.tar.gz" -C "${base_dir}"

    echo "解压缩完成。"
    ;;

  *)
    echo "未知操作: $operation"
    echo "用法: $0 [compress|decompress]"
    exit 1
    ;;
esac
