import os
import sys
import pandas as pd
import exiftool
import argparse
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

import utils.hasher as hasher
import utils.moonphase as moonphase
import utils.ocr as ocr
from utils.labels import ishuman
from utils.labels import get_labels_from_yaml

sys.path.append('../')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(
                    prog='update-cli',
                    description='Update the image archive with new images',
                    epilog='Quills everywhere.')

parser.add_argument('folder', type=str, help='path to folder where the new images are stored')
parser.add_argument('--append', action='store_true', help='append to existing archive.csv', default=False)
args = parser.parse_args()

DATA_FOLDER = './data/'
IMAGE_FOLDER = args.folder
if not os.path.exists(IMAGE_FOLDER):
    print(f"Error: the specified folder {IMAGE_FOLDER} does not exist.")
    sys.exit(1)

folders = os.listdir(IMAGE_FOLDER)

# clean sub-folders list
folders = [f for f in folders if f != '.DS_Store']
folders = [f for f in folders if os.path.isdir(os.path.join(IMAGE_FOLDER, f))]
for f in folders:
    new_name = f.replace(' ', '')
    if new_name != f:
        os.rename(os.path.join(IMAGE_FOLDER, f), os.path.join(IMAGE_FOLDER, new_name))

checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
candidate_labels = get_labels_from_yaml("labels.yaml")

df = pd.DataFrame(columns=['id', 'image_name', 'predicted_animal', 'moon_phase', 'temperature', 'date'])
start_time = pd.Timestamp.now()
for folder in folders:
    files = os.listdir(os.path.join(IMAGE_FOLDER, folder))
    images = [f for f in files if f.endswith('.jpg')]
    image_paths = [os.path.join(IMAGE_FOLDER, folder, path) for path in images]
    
    metadata_dict = {}
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(image_paths)
        for d in metadata:
            try:
                metadata_dict[d["SourceFile"]] = d["EXIF:DateTimeOriginal"]
            except KeyError:
                pass

    for path in tqdm(image_paths, desc=f"Processing images from {folder}"):
        image = Image.open(path)

        predictions = detector(image, candidate_labels=candidate_labels)
        animal_label = predictions[0]["label"]

        if ishuman(animal_label):
            animal_label = "human"

        for pred in predictions:
            if ishuman(pred['label']):
                pred['label'] = 'human'

        seen_labels = set()
        predictions = [pred for pred in predictions if not (pred['label'] in seen_labels or seen_labels.add(pred['label']))]
        
        top_predictions = predictions[:3]

        id = hasher.generate_md5_image_id(image=image)
        
        date = None
        moon_phase = None
        try:
            date = metadata_dict[path]
            date = date.replace(':', '-', 2)
            moon_phase = moonphase.phase(date)
        except KeyError:
            date = ocr.extract_date(image)
            if date:
                moon_phase = moonphase.phase(date)
        except ValueError as e:
            print(f"Error extracting date for image {path}: {e}")
        
        temperature = ocr.extract_temperature(image)

        image_name = path.split('/')[-1]
        
        row = {'id': id, 'image_name': image_name, 'predicted_animal': animal_label, 'moon_phase': moon_phase, 
               'temperature': temperature, 'date': date, 'cam': folder, 'top_predictions': top_predictions}
        
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

elapsed = pd.Timestamp.now() - start_time
print(f"\nUpdate completed.\nElapsed time: {elapsed.total_seconds():.2f}s\nImages processed: {len(df)}")

df.to_csv(os.path.join(DATA_FOLDER, 'update.csv'), index=False)
print(f"\nData saved to {os.path.join(DATA_FOLDER, 'update.csv')}")

if args.append and os.path.exists(os.path.join(DATA_FOLDER, 'archive.csv')):
    aculei_df = pd.read_csv(os.path.join(DATA_FOLDER, 'archive.csv'))
    archive_df = pd.concat([aculei_df, df], ignore_index=True)
    archive_df.to_csv(os.path.join(DATA_FOLDER, 'archive.csv'), index=False)
    print(f"Data appended to {os.path.join(DATA_FOLDER, 'archive.csv')}")
