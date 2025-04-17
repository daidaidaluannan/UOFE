import gwaslab as gl
import pandas as pd
import os
import glob

# 输入文件夹路径
input_dir = "/home/wcy/data/UKB/eye_feature/gwas/main_group/left_flitter_dim64_pheno_result/"
# 输出文件夹路径
output_dir = input_dir + "lead_snp/"

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取所有.fastGWA文件
fastgwa_files = glob.glob(os.path.join(input_dir, "*.fastGWA"))

# 遍历处理每个文件
for file_path in fastgwa_files:
    try:
        # 获取文件名（不带路径和后缀）
        file_name = os.path.basename(file_path).replace('.fastGWA', '')
        
        # 创建Sumstats对象
        mysumstats = gl.Sumstats(file_path,
                                snpid="SNP",
                                chrom="CHR",
                                pos="POS",
                                ea="A1",
                                nea="A2",            
                                n="N",
                                p="P", 
                                build="19",
                                verbose=False)
        
        # 获取lead SNPs
        loci = mysumstats.get_lead(anno=True)
        
        # 如果找到lead SNPs，保存结果
        if not loci.empty:
            output_file = os.path.join(output_dir, f"{file_name}_lead_snps.csv")
            loci.to_csv(output_file, index=False)
            print(f"Successfully processed {file_name} and saved results")
        else:
            print(f"No lead SNPs found for {file_name}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

print("Processing completed!")