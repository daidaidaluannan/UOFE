import os
import subprocess
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="Run GCTA commands for a series of phenotypes.")
parser.add_argument('--bfile_path', type=str,  default='/home/wcy/data/UKB/test_data/eye_gwas/gene/eye_gene', help='Path to the .bfile file')
parser.add_argument('--grm_path', type=str,  default='/home/wcy/data/UKB/test_data/0702/deepnull/sp_grm', help='Path to the GRM sparse file')
parser.add_argument('--pheno_dir', type=str, required=True, help='Directory containing phenotype files')
parser.add_argument('--output_dir', type=str, help='Directory to save the output files')
parser.add_argument('--gcta_path', type=str, default='/home/wcy/python_code/gcta/gcta-1.94.1', help='Path to the GCTA executable')
parser.add_argument('--threads', type=int, default=70, help='Number of threads to use')

args = parser.parse_args()

# 如果用户没有提供 output_dir，则使用默认值
if args.output_dir is None:
    base_dir = os.path.dirname(args.pheno_dir)
    base_name = os.path.basename(args.pheno_dir)
    args.output_dir = os.path.join(base_dir, base_name + '_result')

# 创建输出目录（如果不存在）
os.makedirs(args.output_dir, exist_ok=True)

qcovar_file = os.path.join(args.pheno_dir, "qcovar.txt")
covar_file = os.path.join(args.pheno_dir, "covar.txt")

# 获取 pheno_dir 目录中所有符合条件的文件名
pheno_files = [f for f in os.listdir(args.pheno_dir) if f.startswith('latent_') and f.endswith('.txt')]

# 提取文件名中的数字部分，并找到最大值
max_index = max([int(f.split('_')[1].split('.')[0]) for f in pheno_files])

# 运行 GCTA 命令
for i in range(1, max_index + 1):
    pheno_file = os.path.join(args.pheno_dir, f'latent_{i}.txt')
    output_file = os.path.join(args.output_dir, f'latent_{i}')

    command = [
        args.gcta_path,
        '--bfile', args.bfile_path,
        '--grm-sparse', args.grm_path,
        '--fastGWA-mlm',
        '--pheno', pheno_file,
        '--qcovar', qcovar_file,
        '--covar', covar_file,
        '--thread-num', str(args.threads),
        '--out', output_file
    ]

    print(f"Running: {' '.join(command)}")
    subprocess.run(command)