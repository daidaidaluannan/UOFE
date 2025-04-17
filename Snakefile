# 定义输入和输出文件
MODEL_PATH = "/home/wcy/data/UKB/eye_feature/model/vessel_ViTAutoEncoder.pth"
IMAGE_FOLDER = "/home/wcy/data/UKB/ukb_eye/Results_right/M2/binary_vessel/raw/"
OUTPUT_CSV = "/home/wcy/data/UKB/eye_feature/feature_data/right_vit.csv"
FEATURE_NAME = "right_vessel_vit"
MODEL_TYPE = "VAE_vessel"
LATENT_DIM = 128
GROUP = "main"  # 设置 group 作为全局变量

# 定义 conda 环境
CONDA_ENV = "retfound"

# 目标规则：运行所有任务
rule all:
    input:
        "/home/wcy/data/UKB/eye_feature/cgwas/{group}_group/{feature_name}_result/".format(
            group=GROUP, feature_name=FEATURE_NAME)  # 目标是最后的 cGWAS 输出文件夹

# 提取特征的规则
rule feature_extract:
    input:
        model_path = MODEL_PATH,
        image_folder = IMAGE_FOLDER
    params:
        latent_dim = LATENT_DIM,
        model_type = MODEL_TYPE
    output:
        csv = OUTPUT_CSV
    conda:
        CONDA_ENV
    shell:
        "python feature_extract.py --model_path {input.model_path} --model_type {params.model_type} --image_folder {input.image_folder} --output_csv {output.csv} --latent_dim {params.latent_dim}"

# 预处理 fastGWA 的规则
rule fastGWA_pre:
    input:
        csv = OUTPUT_CSV
    output:
        pheno_dir = directory("/home/wcy/data/UKB/eye_feature/gwas/{group}_group/{feature_name}/".format(
            group=GROUP, feature_name=FEATURE_NAME))
    params:
        feature_name = FEATURE_NAME,
        latent_dim = LATENT_DIM
    conda:
        CONDA_ENV
    shell:
        "python fastGWA_pre.py --eye_file {input.csv} --feature_name {params.feature_name} --latent_dim {params.latent_dim} --save_path {output.pheno_dir}"

# 运行 GCTA 的规则
rule run_gcta:
    input:
        pheno_dir = "/home/wcy/data/UKB/eye_feature/gwas/{group}_group/{feature_name}/".format(
            group=GROUP, feature_name=FEATURE_NAME)
    output:
        input_folder = directory("/home/wcy/data/UKB/eye_feature/gwas/{group}_group/{feature_name}_result/".format(
            group=GROUP, feature_name=FEATURE_NAME))
    conda:
        CONDA_ENV
    shell:
        "python run_gcta.py --pheno_dir {input.pheno_dir} --output_dir {output.input_folder}"

# 预处理 cGWAS 的规则
rule cgwas_pre:
    input:
        input_folder = directory("/home/wcy/data/UKB/eye_feature/gwas/{group}_group/{feature_name}_result/".format(
            group=GROUP, feature_name=FEATURE_NAME))
    output:
        output_folder = directory("/home/wcy/data/UKB/eye_feature/cgwas/{group}_group/{feature_name}_result/".format(
            group=GROUP, feature_name=FEATURE_NAME))
    conda:
        CONDA_ENV
    shell:
        "python cgwas_pre.py --input_folder {input.input_folder} --output_folder {output.output_folder}"
