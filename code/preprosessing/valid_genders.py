import pandas as pd
from pathlib import Path

# Path to the ground truth CSV and dataset directory
CSV_PATH = "/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/images_gender/gender_detection.csv"
IMAGE_ROOT = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/images_gender") 

# Load the CSV file
df = pd.read_csv(CSV_PATH)

# Check for class balance
gender_counts = df['gender'].value_counts()
print("Gender distribution:")
print(gender_counts)

# Check if balanced
if gender_counts.get('woman', 0) != 150 or gender_counts.get('man', 0) != 150:
    print("Dataset is not balanced. It should have 150 woman and 150 man labels.")
else:
    print("Dataset has a balanced gender distribution.")

# Check if all 300 images exist in the dataset folder
missing_files = []

for relative_path in df['file']:
    image_path = IMAGE_ROOT / relative_path
    if not image_path.is_file():
        missing_files.append(str(image_path))

# Report results
if missing_files:
    print(f"Missing {len(missing_files)} images:")
    for file in missing_files:
        print(" -", file)
else:
    print("All 300 images are present and accounted for.")
