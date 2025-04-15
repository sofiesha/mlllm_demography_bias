"""
***************************************************************
*  1. Validate entire dataset (results in 1510 images):
* Script to count and check distribution of image files in the MST-E dataset
*  
*  This script was used to validate the initial dataset by:
*  - Filtering out non-image files (such as videos).
*  - Counting the occurrences of each MST value (MST1, MST2, ..., MST10) across the dataset.
*  - Ensuring only valid image formats (e.g., .jpg, .jpeg, .png, .dng) are considered in the final count.
*
*  The script ensures that the dataset is properly balanced with respect to the image file formats 
*  and the distribution of the MST classes, making the data ready for further processing or analysis.
***************************************************************
"""

import pandas as pd
import os

# Configuration
CSV_PATH = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/mst-e_data/mst-e_image_details.csv"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp", ".svg", ".dng"}
PREPROSESSED_NEW_CSV_PATH = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/mst-e_data/NEW_preprosessed_1510.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

mst1 = 0
mst2 = 0
mst3 = 0
mst4 = 0
mst5 = 0
mst6 = 0
mst7 = 0
mst8 = 0
mst9 = 0
mst10 = 0

# Columns interesting to keep from the original csv (there were other data as well such as lighting and pose)
IMAGE_COLUMN = "image_ID" 
MST_COLUMN = "MST"

# Extract the file extension from each image ID in the IMAGE_COLUMN
file_extensions = df[IMAGE_COLUMN].apply(lambda x: os.path.splitext(str(x).strip().lower())[1])

# Get the unique file extensions
unique_extensions = file_extensions.unique()

# Print the unique extensions
print("Unique file extensions found in the image_ID column:")
for ext in unique_extensions:
    print(ext)

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    
    image_name = row[IMAGE_COLUMN]
    mst_value = row[MST_COLUMN]

    if image_name.lower().endswith(tuple(VALID_EXTENSIONS)):
        if mst_value == 1:
            mst1+=1
            #print(f"Another entry for mst1:{mst1}")
        elif mst_value == 2:
            mst2+=1
            #print(f"Another entry for mst2:{mst2}")
        elif mst_value == 3:
            mst3+=1
            #print(f"Another entry for mst2:{mst3}")
        elif mst_value == 4:
            mst4+=1
            #print(f"Another entry for mst2:{mst4}")
        elif mst_value == 5:
            mst5+=1
            #print(f"Another entry for mst2:{mst5}")
        elif mst_value== 6:
            mst6+=1
            #print(f"Another entry for mst2:{mst6}")
        elif mst_value == 7:
            mst7+=1
            #print(f"Another entry for mst2:{mst7}")
        elif mst_value == 8:
            mst8+=1
            #print(f"Another entry for mst2:{mst8}")
        elif mst_value == 9:
            mst9+=1
            #print(f"Another entry for mst2:{mst9}")
        else:
            mst10+=1     
            #print(f"Another entry for mst2:{mst10}")
        
print(f"MST count:\n MST1 = {mst1} \n MST2 = {mst2} \n MST3 = {mst3} \n MST4 = {mst4} \n MST5 = {mst5} \n MST6 = {mst6} \n MST7 = {mst7} \n MST8 = {mst8} \n MST9 = {mst9} \n MST 10 = {mst10}")

# Calculate the total count of valid images
total_valid_images = mst1 + mst2 + mst3 + mst4 + mst5 + mst6 + mst7 + mst8 + mst9 + mst10

# Print total count
print(f"\nTotal valid image count: {total_valid_images}")
    
# Initialize a list to hold rows that will be written to the new CSV
filtered_rows = []

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    image_name = row[IMAGE_COLUMN]
    mst_value = row[MST_COLUMN]

    # Check if the image file has a valid extension
    if image_name.lower().endswith(tuple(VALID_EXTENSIONS)):
        # Append valid rows to the list
        filtered_rows.append(row)

# Create a new DataFrame from the filtered rows
filtered_df = pd.DataFrame(filtered_rows)

# Save the filtered dataset to a new CSV file
filtered_df.to_csv(PREPROSESSED_NEW_CSV_PATH, index=False)

# Print confirmation message
print(f"\nFiltered dataset saved to: {PREPROSESSED_NEW_CSV_PATH}")


#****************************************************************
## DOING IT ONE MORE TIME TO VALIDATE THE NEW CSV FILE AS WELL

# Load CSV
df = pd.read_csv(PREPROSESSED_NEW_CSV_PATH)

mst1 = 0
mst2 = 0
mst3 = 0
mst4 = 0
mst5 = 0
mst6 = 0
mst7 = 0
mst8 = 0
mst9 = 0
mst10 = 0

IMAGE_COLUMN = "image_ID" 
MST_COLUMN = "MST"
SUBJECT_NAME = "subject_name" 

# Extract the file extension from each image ID in the IMAGE_COLUMN
file_extensions = df[IMAGE_COLUMN].apply(lambda x: os.path.splitext(str(x).strip().lower())[1])

# Get the unique file extensions
unique_extensions = file_extensions.unique()

# Print the unique extensions
print("Unique file extensions found in the image_ID column:")
for ext in unique_extensions:
    print(ext)

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    
    image_name = row[IMAGE_COLUMN]
    mst_value = row[MST_COLUMN]

    if image_name.lower().endswith(tuple(VALID_EXTENSIONS)):
        if mst_value == 1:
            mst1+=1
            #print(f"Another entry for mst1:{mst1}")
        elif mst_value == 2:
            mst2+=1
            #print(f"Another entry for mst2:{mst2}")
        elif mst_value == 3:
            mst3+=1
            #print(f"Another entry for mst2:{mst3}")
        elif mst_value == 4:
            mst4+=1
            #print(f"Another entry for mst2:{mst4}")
        elif mst_value == 5:
            mst5+=1
            #print(f"Another entry for mst2:{mst5}")
        elif mst_value== 6:
            mst6+=1
            #print(f"Another entry for mst2:{mst6}")
        elif mst_value == 7:
            mst7+=1
            #print(f"Another entry for mst2:{mst7}")
        elif mst_value == 8:
            mst8+=1
            #print(f"Another entry for mst2:{mst8}")
        elif mst_value == 9:
            mst9+=1
            #print(f"Another entry for mst2:{mst9}")
        else:
            mst10+=1     
            #print(f"Another entry for mst2:{mst10}")
        
print(f"MST count in NEW CSV:\n MST1 = {mst1} \n MST2 = {mst2} \n MST3 = {mst3} \n MST4 = {mst4} \n MST5 = {mst5} \n MST6 = {mst6} \n MST7 = {mst7} \n MST8 = {mst8} \n MST9 = {mst9} \n MST 10 = {mst10}")
    
# Calculate the total count of valid images
total_valid_images_NEW = mst1 + mst2 + mst3 + mst4 + mst5 + mst6 + mst7 + mst8 + mst9 + mst10

# Print total count
print(f"\nTotal valid image count: {total_valid_images_NEW}")

if total_valid_images != total_valid_images_NEW:
    print("DATASET NOT VALIDATED")
else:
    print("DATASET IS VALID!!!!")
    

#*******************************************
#CONTINUING WITH REMOVING POTENTIAL DUPLICATES
    
# Load the cleansed CSV
df = pd.read_csv(PREPROSESSED_NEW_CSV_PATH)

# Remove duplicates based on the 'image_ID' column
df_cleaned = df.drop_duplicates(subset='image_ID')

# Save the cleaned CSV back to a new file
CLEANSED_CSV_CLEANED_PATH = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/data/skin_tone/preprosessed_initial_MST_full_1510.csv"
df_cleaned.to_csv(CLEANSED_CSV_CLEANED_PATH, index=False)

print(f"Duplicates removed. Cleaned CSV saved at {CLEANSED_CSV_CLEANED_PATH}")



#************************************
# ONE LAST TIME TO SEE HOW MANY ENTRIES NOW (ANY DUPLICATES?)

# Load CSV
df = pd.read_csv(PREPROSESSED_NEW_CSV_PATH)

mst1 = 0
mst2 = 0
mst3 = 0
mst4 = 0
mst5 = 0
mst6 = 0
mst7 = 0
mst8 = 0
mst9 = 0
mst10 = 0

IMAGE_COLUMN = "image_ID" 
MST_COLUMN = "MST"
SUBJECT_NAME = "subject_name" 

# Extract the file extension from each image ID in the IMAGE_COLUMN
file_extensions = df[IMAGE_COLUMN].apply(lambda x: os.path.splitext(str(x).strip().lower())[1])

# Get the unique file extensions
unique_extensions = file_extensions.unique()

# Print the unique extensions
print("Unique file extensions found in the image_ID column:")
for ext in unique_extensions:
    print(ext)

# Iterate through the rows of the dataframe
for index, row in df.iterrows():
    
    image_name = row[IMAGE_COLUMN]
    mst_value = row[MST_COLUMN]

    if image_name.lower().endswith(tuple(VALID_EXTENSIONS)):
        if mst_value == 1:
            mst1+=1
            #print(f"Another entry for mst1:{mst1}")
        elif mst_value == 2:
            mst2+=1
            #print(f"Another entry for mst2:{mst2}")
        elif mst_value == 3:
            mst3+=1
            #print(f"Another entry for mst2:{mst3}")
        elif mst_value == 4:
            mst4+=1
            #print(f"Another entry for mst2:{mst4}")
        elif mst_value == 5:
            mst5+=1
            #print(f"Another entry for mst2:{mst5}")
        elif mst_value== 6:
            mst6+=1
            #print(f"Another entry for mst2:{mst6}")
        elif mst_value == 7:
            mst7+=1
            #print(f"Another entry for mst2:{mst7}")
        elif mst_value == 8:
            mst8+=1
            #print(f"Another entry for mst2:{mst8}")
        elif mst_value == 9:
            mst9+=1
            #print(f"Another entry for mst2:{mst9}")
        else:
            mst10+=1     
            #print(f"Another entry for mst2:{mst10}")
        
print(f"MST count in NEW CSV:\n MST1 = {mst1} \n MST2 = {mst2} \n MST3 = {mst3} \n MST4 = {mst4} \n MST5 = {mst5} \n MST6 = {mst6} \n MST7 = {mst7} \n MST8 = {mst8} \n MST9 = {mst9} \n MST 10 = {mst10}")
    
# Calculate the total count of valid images
total_images_NEW = mst1 + mst2 + mst3 + mst4 + mst5 + mst6 + mst7 + mst8 + mst9 + mst10

# Print total count
print(f"\nNEW total image count: {total_images_NEW}")

if total_valid_images_NEW != total_images_NEW:
    total = total_valid_images - total_valid_images_NEW
    print(f"Nr. of duplicates removed: {total}")
else:
    print("No duplicates were found")
