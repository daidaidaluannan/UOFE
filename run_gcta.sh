#!/bin/bash

# 检查输入参数数量
if [ "$#" -ne 4 ]; then
    echo "Usage: $0  <bfile_path> <grm_path> <pheno_dir> <output_dir>"
    exit 1
fi

# 获取输入参数

BFILE_PATH=$1
GRM_PATH=$2
PHENO_DIR=$3
OUTPUT_DIR=$4

# 其他固定文件路径
QCOVAR_FILE="$PHENO_DIR/qcovar.txt"
COVAR_FILE="$PHENO_DIR/covar.txt"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 循环执行 GCTA 命令
for i in {1..128}; do
    PHENO_FILE="$PHENO_DIR/latent_${i}.txt"
    OUTPUT_FILE="$OUTPUT_DIR/latent_${i}"

    /home/wcy/python_code/gcta/gcta-1.94.1 --bfile "$BFILE_PATH" \
                           --grm-sparse "$GRM_PATH" \
                           --fastGWA-mlm \
                           --pheno "$PHENO_FILE" \
                           --qcovar "$QCOVAR_FILE" \
                           --covar "$COVAR_FILE" \
                           --thread-num 70 \
                           --out "$OUTPUT_FILE"
done
