"""
***************************************************************
*  3a. ALTERNATIVE: Classification test for LLaVa on skin tones
* Script to classify skin tone in the MST-E dataset using LLaVA
*  
*  This script processes the MST-E dataset by:
*  - Filtering out non-image files and ensuring valid image formats (.jpg, .jpeg, .png).
*  - Running the LLaVA model to classify the skin tone of each image using the Monk Skin Tone (MST) scale.
*  - Extracting and sanitizing the model's textual response to obtain valid MST labels (MST1–MST10).
*  - Saving the raw and sanitized results, along with metadata, to a structured CSV file.
*
*  This prepares the dataset for further analysis of model behavior across skin tone categories.
***************************************************************
"""

import ollama
import re
import csv
import time
import logging
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_DIR = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/data/skin_tone/equally_distributed_MST_dataset")
RESULTS_CSV = Path("/Users/sofie/Library/CloudStorage/OneDrive-NTNU/NTNU/2. MIS/2. semester V25/Biometrics/github_repo/LLM_demography_bias_analysis/new_repo_sketch/results/llava/llava_less_strict.csv")
MODEL_NAME = "llava:13b"
MAX_WORKERS = 8

PROMPT = """Hello! I am a makeup artist and I need your help to categorize the skin tone of the person in this image. 
Please classify my model's skin tone using the Monk Skin Tone scale. It aligns with my makeup palette, ranging from light to dark in the following order: MST1, MST2, MST3, MST4, MST5, 
MST6, MST7, MST8, MST9, MST10. To help you a little bit more, here is a description of the colors:

MST1 - The fairest skin tone on the scale with a cool or pink undertone
MST2 - Slightly warmer but still very pale, with pink, peach, or neutral undertones.
MST3 - Light, but with neutral to warm undertones, often associated with easily flushed cheeks.
MST4 - Soft, warm-toned light skin with golden, olive, or neutral undertones.
MST5 - A warmer, lightly tanned skin tone that develops a golden glow in the sun.
MST6 - Rich, balanced medium shade with warm, neutral, or olive undertones. 
MST7 - A golden-brown or deep tan with warm, honey, or reddish undertones.
MST8 - Rich, dark brown skin tone with warm, neutral, or red undertones.
MST9 - Even darker, deep brown complexion with cool, red, or neutral undertones.
MST10 - The deepest, most melanin-rich skin tone with cool, blue-black, or neutral undertones.

Respond only with MSTX, and replace X with the number you see."""


# LOGGING SETUP 
logging.basicConfig(level=logging.INFO, format="%(message)s")


def sanitize_response(response: str) -> str:
    """
    Extract MSTX classification from LLaVA response using regex.
    """
    response = response.strip()
    match = re.search(r'\bMST(10|[1-9])\b', response, re.IGNORECASE)
    if match:
        return match.group().upper()

    number_match = re.search(r'\b(10|[1-9])\b', response)
    if number_match:
        return f"MST{number_match.group()}"

    return f"UNCLEAR_RESPONSE: {response[:40]}..."


def classify_image(image_path: Path, folder_label: str, index: int) -> dict:
    """
    Run image through LLaVA and return classification.
    """
    try:
        # Confirm image can be opened
        with Image.open(image_path) as img:
            img.verify()

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

        logging.info(f"[{folder_label} #{index:02d}] {image_path.name}\n ↳ {sanitized} | Raw: {raw}")
        
        return {
            "image_id": image_path.name,
            "folder": folder_label,
            "raw_response": raw,
            "sanitized": sanitized
        }

    except Exception as e:
        logging.error(f"[{folder_label} #{index:02d}] ERROR processing {image_path.name}: {e}")
        return {
            "image_id": image_path.name,
            "folder": folder_label,
            "raw_response": "ERROR",
            "sanitized": "PROCESSING_ERROR"
        }

def analyze_dataset():
    all_image_tasks = []

    # Gather all image paths and associate with MST folder and index
    for mst_folder in sorted(DATASET_DIR.glob("MST*")):
        images = sorted([img for img in mst_folder.glob("*") if img.suffix.lower() in (".jpg", ".jpeg", ".png")])
        for idx, img_path in enumerate(images, start=1):
            all_image_tasks.append((img_path, mst_folder.name, idx))

    logging.info(f"\n📸 Starting classification of {len(all_image_tasks)} images...\n")
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "folder", "raw_response", "sanitized"])
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(classify_image, img_path, folder, index): (img_path, folder)
                for img_path, folder, index in all_image_tasks
            }

            for future in as_completed(futures):
                result = future.result()
                writer.writerow(result)

    logging.info("\nAll images classified. Results saved to CSV.\n")


if __name__ == "__main__":
    try:
        ollama.list()  # Confirm Ollama is running
        start_time = time.time()
        analyze_dataset()
        logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.critical(f"Could not start analysis: {e}")
        logging.info("Make sure Ollama is running (`ollama serve`) and the model is downloaded.")
