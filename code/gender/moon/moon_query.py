"""
***************************************************************
*  Query Moondream for gender classification:
*
*  This script queries moondream to classify gender based on face images by:
*  - Loopng through the dataset of 300 images
*  - Sending it to moon along with a prompt, and threads the process
*  - Logging both a raw and sanitized ouput in the csv file along with ground truth
*
*  This provides a rapid automated query for moon on gender recognition.
***************************************************************
"""

import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ollama 


GROUND_TRUTH_CSV = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/images_gender/gender_detection.csv")
IMAGE_ROOT = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/images_gender")
OUTPUT_CSV = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/results/gender/moon/gender_classification.csv")
MODEL_NAME = "moondream:latest"
PROMPT = """Hello! I am a makeup artist and I need your help to categorize the gender of the person in this image. Is this person a man or a woman?"""
GENDER_REGEX = re.compile(r"\b(man|woman|male|female|boy|girl)\b", re.IGNORECASE)

# Load ground truth
df = pd.read_csv(GROUND_TRUTH_CSV)
df["raw_response"] = ""
df["sanitized"] = ""

# Sanitization of raw response
def sanitize_response(raw: str) -> str:
    match = GENDER_REGEX.search(raw)
    return match.group(0).lower() if match else "uncertain"

#Moondream classification function
def classify_image(image_path: Path, folder_label: str, index: int) -> dict:
    """
    Run image through Moondream and return classification.
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Just to make sure image is readable

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [str(image_path)]
            }],
            options={"temperature": 0.5}
        )

        raw = response["message"]["content"].strip()
        sanitized = sanitize_response(raw)

        print(f"[{folder_label.upper()}] Image {index}: GT={df.at[index, 'gender']} | Predicted={sanitized} | Raw={raw}")

        return {"index": index, "raw": raw, "sanitized": sanitized}

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {"index": index, "raw": "ERROR", "sanitized": "ERROR"}

# Multithreaded image processing for speed
max_workers = min(8, os.cpu_count())
futures = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for idx, row in df.iterrows():
        image_path = IMAGE_ROOT / row["file"]
        split_label = row["split"]
        futures.append(executor.submit(classify_image, image_path, split_label, idx))

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
        result = future.result()
        df.at[result["index"], "raw_response"] = result["raw"]
        df.at[result["index"], "sanitized"] = result["sanitized"]

# Save results to csv
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFinished. Results saved to: {OUTPUT_CSV}")