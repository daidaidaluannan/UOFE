from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm


input_folder = '/home/wcy/data/UKB/ukb_eye/Results_right/M1/Good_quality/'
output_folder = '/home/wcy/data/UKB/ukb_eye/Results_right/M1/right_flitter/'


os.makedirs(output_folder, exist_ok=True)


image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpeg')]


for image_file in tqdm(image_files):

    input_image_path = os.path.join(input_folder, image_file)
    image = Image.open(input_image_path)
    image = image.resize((256, 256))


    image_np = np.array(image)


    filtered_image = cv2.medianBlur(image_np, 15)


    filtered_image_pil = Image.fromarray(filtered_image)


    output_image_path = os.path.join(output_folder, image_file)
    filtered_image_pil.save(output_image_path)

    #print(f'Processed and saved: {output_image_path}')

print("Batch processing completed.")
