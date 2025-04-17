import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)



parser = argparse.ArgumentParser(description="Process UKB eye feature data.")
parser.add_argument("--input_folder", type=str, required=True,help="Path to the fastGWA result folder.")
parser.add_argument("--output_folder", type=str, required=True,help="Path to the cgwas folder.")
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder

base_name = os.path.basename(input_folder)
output_folder = os.path.join(output_folder, base_name)

os.makedirs(output_folder, exist_ok=True)


column2 = ['BETA','P']

gwas_files = [f for f in os.listdir(input_folder) if f.endswith('.fastGWA')]
print(output_folder)
for gwas_file in tqdm(gwas_files):
    target = gwas_file.replace('.fastGWA', '.assoc')
    input_path = os.path.join(input_folder, gwas_file)
    output_path = os.path.join(output_folder, target)

    gwas = pd.read_csv(input_path,sep = '\t',usecols=column2)
    gwas.to_csv(output_path,'\t',index=False, header=True)