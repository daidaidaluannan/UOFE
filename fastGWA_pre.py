import csv
import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process UKB eye feature data.")
parser.add_argument("--eye_file", type=str, required=True,help="Path to the eye feature CSV file.")
parser.add_argument("--feature_name", type=str, required=True,help="Name of the feature.")
parser.add_argument('--latent_dim', type=int, required=True, help='Dimensionality of the latent space.')

parser.add_argument("--save_path", type=str, default= "/home/wcy/data/UKB/eye_feature/gwas/main_group/" ,
                    help="Path to save the main group results.")

parser.add_argument("--validation_path", type=str, default= "/home/wcy/data/UKB/eye_feature/gwas/validation_group/" ,
                    help="Path to save the validation group results.")

parser.add_argument("--all_data_file", type=str, default= "/home/wcy/data/UKB/ukb47147.csv" 
                    ,help="Path to the all data CSV file.")

parser.add_argument("--data_table", type=str, default= "/home/wcy/data/UKB/test_data/0620_gwas/eye_gene/data_table.csv"
                    ,help="Path to the data table CSV file.")

parser.add_argument("--pca_file", type=str, default= "/home/wcy/data/UKB/test_data/0620_gwas/eye_gene/imputation/step_3/eye_pca.csv"
                    ,help="Path to the PCA CSV file.")

parser.add_argument("--disease_file", type=str, default= "/home/wcy/python_code/ICDBioAssign-master/Templates/eye_disease/eye_disease.csv"
                    ,help="Path to the disease CSV file.")
    
args = parser.parse_args()


eye_file = args.eye_file
feature_name = args.feature_name
save_dir = args.save_path
validation_path = args.validation_path
all_data_file = args.all_data_file
data_table = args.data_table
pca_file = args.pca_file
disease_file = args.disease_file



#save_dir = save_path + feature_name + "/"
validation_dir = validation_path + feature_name + "/"


# 创建保存目录
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(validation_dir):
    os.makedirs(validation_dir)

# 读取要提取的列名
with open(data_table, 'r', encoding='utf-8-sig') as csvfile: 
    reader = csv.reader(csvfile)
    column1 = [row[0] for row in reader]
    print('提取的列名称：', column1)

disease_data = pd.read_csv(disease_file)
all_data = pd.read_csv(all_data_file, usecols=column1)
all_data = all_data.dropna(axis=0, how='any', subset=column1, inplace=False) 

print('提取前样本数:', all_data.shape[0])
eye_data = pd.read_csv(eye_file, sep=',')
pheno_raw = pd.merge(all_data, eye_data, on='eid')
print('提取后样本数:', pheno_raw.shape[0])

# 筛选掉 Retinal 和 Glaucoma 为 1 的数据
pheno_raw = pd.merge(pheno_raw, disease_data[['eid', 'Retinal', 'Glaucoma']], on='eid')
pheno_raw = pheno_raw[(pheno_raw['Retinal'] != 1) & (pheno_raw['Glaucoma'] != 1)]
print("筛选后的样本数:", pheno_raw.shape[0])

# 按照 "21000-0.0" 列进行数据分割
main_cohort = pheno_raw[pheno_raw['21000-0.0'] == 1001]
validation_cohort = pheno_raw[pheno_raw['21000-0.0'].isin([1002, 1003])]

# 处理 PCA 数据并合并到主队列和验证队列
pca = pd.read_csv(pca_file)

# 主队列处理
main_use = pd.merge(pca, main_cohort, on='eid')
main_use.insert(0, 'FID', main_use.iloc[:, 0])
main_use.rename(columns={'eid': 'IID'}, inplace=True)

# 验证队列处理
validation_use = pd.merge(pca, validation_cohort, on='eid')
validation_use.insert(0, 'FID', validation_use.iloc[:, 0])
validation_use.rename(columns={'eid': 'IID'}, inplace=True)

# 定量协变量（qcovar）和分类协变量（covar）的保存函数
def save_covariates(data, folder):
    qcovar = data.loc[:, ['FID', 'IID', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10',
                          '21001-0.0', '21003-0.0']]
    covar = data.loc[:, ['FID', 'IID', '31-0.0', '20116-0.0', '20117-0.0']]
    qcovar.to_csv(os.path.join(folder, 'qcovar.txt'), sep='\t', index=False, header=False)
    covar.to_csv(os.path.join(folder, 'covar.txt'), sep='\t', index=False, header=False)

# 保存主队列的协变量
save_covariates(main_use, save_dir)

# 保存验证队列的协变量
save_covariates(validation_use, validation_dir)



# 保存潜在特征数据
def save_latent_features(data, folder):
    latent_columns = [f'latent_{i}' for i in range(1, args.latent_dim + 1)]
    for eye_column_name in latent_columns:
        pheno = data.loc[:, ['FID', 'IID', eye_column_name]]
        file_name = eye_column_name + '.txt'
        pheno.to_csv(os.path.join(folder, file_name), sep='\t', index=False, header=True)

# 保存主队列的潜在特征
save_latent_features(main_use, save_dir)

# 保存验证队列的潜在特征
save_latent_features(validation_use, validation_dir)
