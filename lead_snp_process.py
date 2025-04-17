import os
import pandas as pd
import gwaslab as gl

# 定义输入文件夹和输出文件路径
input_folder = "/home/wcy/data/UKB/eye_feature/gwas/main_group/left_flitter_dim64_pheno_result/"
output_csv =  input_folder + "lead_snp_results.csv"

# 批处理文件夹中的所有以 .fastGWA 为后缀的文件
file_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".fastGWA")]

# 存储所有文件的结果
all_lead_snps = []

# 遍历每个文件进行分析
for file_path in file_list:
    print(f"Processing file: {file_path}")
    
    # 加载 GWAS 数据
    mysumstats = gl.Sumstats(
        file_path,
        snpid="SNP",
        chrom="CHR",
        pos="POS",
        ea="A1",
        nea="A2",
        beta="BETA",
        se="SE",
        p="P",
        build="19",
        verbose=False
    )
    
    # 提取 lead SNP 信息
    loci_data = mysumstats.get_lead(anno=True,sig_level=1e-10)
    
    # 添加文件来源列
    loci_data["Source_File"] = os.path.basename(file_path)
    
    # 存储结果
    all_lead_snps.append(loci_data)

# 合并所有结果为一个 DataFrame
result_df = pd.concat(all_lead_snps, ignore_index=True)

# 将结果保存为 CSV 文件
result_df.to_csv(output_csv, index=False)
print(f"Lead SNP results saved to: {output_csv}")
