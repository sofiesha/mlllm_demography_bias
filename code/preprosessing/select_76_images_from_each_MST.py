"""
***************************************************************
*  2. Select equally distributed dataset:
* Script to select 76 images from each MST class and organize 
*     them into subfolders for a balanced dataset
*  
*  This script prepares an equally distributed subset of the MST-E 
*  dataset by:
*  - Creating 10 subfolders (MST1 to MST10) within the output directory.
*  - Iterating through a cleansed CSV file containing valid image entries.
*  - Selecting the first 76 images for each MST class.
*  - Locating the images across nested source directories using regex.
*  - Copying the selected images into their corresponding MST subfolders.
*
*  The result is a structured and balanced dataset with exactly 76 
*  images per MST class.
***************************************************************
"""

import shutil
import os
import pandas as pd
import re
from pathlib import Path

# Configuration
SOURCE_DIR = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/mst-e_data"  # Base directory where the images are stored (including subdirectories)
DEST_DIR = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/data/skin_tone/equally_distributed_MST_dataset"  # Folder to store the categorized images
CLEANSED_CSV_PATH = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/data/skin_tone/preprosessed_initial_MST_full_1510.csv"  # The new cleansed CSV with valid images
IMAGE_COL = 'image_ID'                                         # Column in CSV with image names
MST_COL = 'MST'                                                # Column in CSV with MST class
SUBJECT_REGEX = r"subject_\d+"                                 # Regex pattern for subfolders
NUM_IMAGES = 76                                                # Number of images per MST class

#STEP 1: Load CSV
df = pd.read_csv(CLEANSED_CSV_PATH)

#STEP 2: Create MST folders
for i in range(1, 11):
    mst_folder = os.path.join(DEST_DIR, f"MST{i}")
    os.makedirs(mst_folder, exist_ok=True)

#STEP 3: Index all image paths under source directory
print("Indexing image files...")
image_paths = {}
for root, dirs, files in os.walk(SOURCE_DIR):
    if re.search(SUBJECT_REGEX, root):
        for file in files:
            image_paths[file] = os.path.join(root, file)

#STEP 4: Copy images for each MST
for mst_class in sorted(df[MST_COL].unique()):
    mst_df = df[df[MST_COL] == mst_class].head(NUM_IMAGES)
    dest_folder = os.path.join(DEST_DIR, f"MST{int(mst_class)}")

    for _, row in mst_df.iterrows():
        image_name = row[IMAGE_COL]
        if image_name in image_paths:
            shutil.copy(image_paths[image_name], os.path.join(dest_folder, image_name))
        else:
            print(f"Image not found: {image_name}")

print("Done! Each MST folder should now have exactly 76 images.")
