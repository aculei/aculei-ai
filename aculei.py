import sys
import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import utils.hasher as hasher
import utils.moonphase as moonphase
import utils.ocr as ocr
import utils.labels as labels_util
import exiftool
from transformers import pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"
checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

def ishuman(label):
    return label in ["human", "girl", "man", "woman", "old woman", "boy", "old man", "person", "people"]

def save_to_csv(results_list, output_path):
    if not results_list:
        return
    df = pd.DataFrame(results_list)
    file_exists = os.path.isfile(output_path)
    df.to_csv(output_path, mode='a', index=False, header=not file_exists)
    print(f"\n[Disk Sync] Appended {len(results_list)} rows to {output_path}")

def main(selected_dir, labels_file="labels.yaml", out="aculei.csv"):
    image_folder = selected_dir
    labels = labels_util.get_labels_from_yaml(labels_file)
    if not os.path.exists(image_folder):
        print(f"Error: The directory {image_folder} does not exist.")
        return
    if os.path.exists(out):
        print(f"Warning: {out} already exists. New data will be appended to it.")

    image_data = []
    all_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.startswith('.') or not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, image_folder)
            macro_dir = relative_path.split(os.sep)[0]
            if macro_dir == ".": macro_dir = "root"

            image_data.append({"path": full_path, "camera": macro_dir})
            all_paths.append(full_path)

    metadata_dict = {}
    print(f"Extracting EXIF metadata for {len(all_paths)} images...")
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(all_paths)
        for d in metadata:
            if "EXIF:DateTimeOriginal" in d:
                metadata_dict[d["SourceFile"]] = d["EXIF:DateTimeOriginal"]

    results = []
    for i, item in enumerate(tqdm(image_data, desc="Processing images")):
        img_path = item["path"]
        camera_name = item["camera"]
        
        try:
            image = Image.open(img_path)
            predictions = detector(image, candidate_labels=labels)

            for pred in predictions:
                if ishuman(pred['label']):
                    pred['label'] = 'human'
            
            seen_labels = set()
            predictions = [pred for pred in predictions if not (pred['label'] in seen_labels or seen_labels.add(pred['label']))]
            
            animal_label = predictions[0]["label"]
            top_predictions = predictions[:3]
            img_id = hasher.generate_md5_image_id(image=image)

            date = None
            moon_phase = None
            
            if img_path in metadata_dict:
                date = metadata_dict[img_path]
                date = date.replace(':', '-', 2)
                moon_phase = moonphase.phase(date)
            else:
                date = ocr.extract_date(image)
                if date:
                    moon_phase = moonphase.phase(date)
            
            temperature = ocr.extract_temperature(image)
            image_name = os.path.basename(img_path)

            results.append({
                'id': img_id, 
                'image_name': image_name, 
                'predicted_animal': animal_label, 
                'moon_phase': moon_phase, 
                'temperature': temperature, 
                'date': date, 
                'cam': camera_name, 
                'top_predictions': top_predictions
            })

            if (i + 1) % 1000 == 0:
                save_to_csv(results, out)
                results = []

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    if results:
        save_to_csv(results, out)
    
    print(f"Processing complete. All data saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a specific directory.")
    parser.add_argument('--dir', type=str, required=True, help='The directory name to process')
    parser.add_argument('--labels', type=str, default='labels.yaml', help='The labels YAML file')
    parser.add_argument('--output', type=str, default='aculei.csv', help='The output CSV file')

    args = parser.parse_args()
    main(args.dir, args.labels, args.output)