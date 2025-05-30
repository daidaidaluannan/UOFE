import csv
import os
import numpy as np
import pandas as pd

cor_path = "/home/wcy/data/UKB/eye_feature/disease_cor/cor_eye_gwas/left_vessel_dim128_cor/left_vessel_dim128_cor.csv"
ldsc_path1 = "/home/wcy/data/UKB/eye_feature/gwas/main_group/left_vessel_dim128_pheno_result/ldsc/"
ldsc_path2 = "/home/wcy/data/UKB/test_data/eye_gwas/ldsc_data/"

result_path = "/home/wcy/data/UKB/eye_feature/disease_cor/cor_eye_gwas/left_vessel_dim128_cor/"

ref_path = "/home/wcy/python_code/ldsc/tool_data/eur_w_ld_chr/"
w_path = "/home/wcy/python_code/ldsc/tool_data/eur_w_ld_chr/"

cor_file = pd.read_csv(cor_path,',')

for index, row in cor_file.iterrows():
    col1 = row['Var1']
    col2 = row['Var2']
    file_name1 = ldsc_path1 + col1
    file_name2 = ldsc_path2 +  col2
    rg_name = file_name1 + ',' + file_name2

    # 去掉后缀
    col1_base = os.path.splitext(os.path.splitext(col1)[0])[0]  # 去掉 .sumstats.gz
    col2_base = os.path.splitext(os.path.splitext(col2)[0])[0]  # 去掉 .sumstats.gz
    result_name = result_path + col1_base + "-" + col2_base

    cmd = f"~/python_code/ldsc/ldsc.py --out {result_name} --rg {rg_name} --ref-ld-chr {ref_path} --w-ld-chr {ref_path}"
    os.system(cmd)

    with open(result_name+'.log', "r") as file:
        lines = file.readlines()
    
    genetic_corr = None
    genetic_se =None
    z_score = None
    p_value = None

    # 遍历每一行并查找目标值
    for line in lines:
        if line.startswith("Genetic Correlation:"):
            genetic_corr_raw = line.split(":")[1].strip()
            genetic_corr = genetic_corr_raw.split(" ")[0].strip()
            genetic_se = genetic_corr_raw.split(" ")[1].strip()
            genetic_se = genetic_se.replace("(", "").replace(")", "")
        elif line.startswith("Z-score"):
            z_score = line.split(":")[1].strip()
        elif line.startswith("P"):
            p_value = line.split(":")[1].strip()
    
    cor_file.at[index, 'Genetic_Correlation'] = genetic_corr
    cor_file.at[index, 'Genetic_Se'] = genetic_se
    cor_file.at[index, 'Z-score'] = z_score
    cor_file.at[index, 'P_gene'] = p_value

cor_file.to_csv(result_path + "gene_cor.csv",'\t',index=False, header=True)
       