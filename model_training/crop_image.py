# crop_test_with_margin.py

import os
import cv2
import pandas as pd
import random
import glob
import numpy as np

# ----------- SETTINGS -------------
BASE = 'data'
IMG_DIR = os.path.join(BASE, 'datasets')
TRAIN_CSV = os.path.join(BASE, 'train_labels_filtered.csv')
TEST_CSV = os.path.join(BASE, 'test_labels_filtered.csv')

OUTPUT_TRAIN = os.path.join(BASE, 'train')
OUTPUT_TEST = os.path.join(BASE, 'test')

# Margin in pixels
MARGIN = 110

# Create output dirs for both splits
for out_dir in [OUTPUT_TRAIN, OUTPUT_TEST]:
    for cls in ['stenosis', 'normal']:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

def find_alternative_file(img_dir, base_name):
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in exts:
        alt_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(alt_path):
            return alt_path
    matches = glob.glob(os.path.join(img_dir, base_name + '.*'))
    if matches:
        return matches[0]
    return None

def process_split(df, output_dir, label):
    missing_files = []
    alt_found = []
    log_path = os.path.join(output_dir, 'log.txt')
    with open(log_path, 'w') as logf:
        for i, row in df.iterrows():
            img_path = os.path.join(IMG_DIR, row['filename'])
            if not os.path.exists(img_path):
                base_name, _ = os.path.splitext(row['filename'])
                alt = find_alternative_file(IMG_DIR, base_name)
                if alt:
                    alt_found.append((img_path, alt))
                    img_path = alt
                else:
                    missing_files.append(img_path)
                    print(f"Missing: {img_path}")
                    continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_path}")
                missing_files.append(img_path)
                continue
            h, w = img.shape[:2]
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            # Add margin
            x1m = max(0, x1 - MARGIN)
            y1m = max(0, y1 - MARGIN)
            x2m = min(w, x2 + MARGIN)
            y2m = min(h, y2 + MARGIN)
            # Positive crop
            pos_crop = img[y1m:y2m, x1m:x2m]
            pos_crop = cv2.resize(pos_crop, (224, 224), interpolation=cv2.INTER_AREA)
            pos_out = os.path.join(output_dir, 'stenosis', f'{i}_pos.jpg')
            cv2.imwrite(pos_out, pos_crop)
            # Log info
            logf.write(f"{row['filename']}\n")
            logf.write(f"  Original bbox: ({x1}, {y1}), ({x2}, {y2})\n")
            logf.write(f"  Margin bbox:   ({x1m}, {y1m}), ({x2m}, {y2m})\n")
            logf.write(f"  Cropped shape: {pos_crop.shape}\n\n")
            # Negative crop: random region away from bbox
            tries = 0
            rw = x2m - x1m
            rh = y2m - y1m
            while tries < 50:
                rx = random.randint(0, w - rw)
                ry = random.randint(0, h - rh)
                if rx > x2m or ry > y2m or (rx + rw < x1m) or (ry + rh < y1m):
                    neg_crop = img[ry:ry+rh, rx:rx+rw]
                    neg_crop = cv2.resize(neg_crop, (224, 224), interpolation=cv2.INTER_AREA)
                    neg_out = os.path.join(output_dir, 'normal', f'{i}_neg.jpg')
                    cv2.imwrite(neg_out, neg_crop)
                    break
                tries += 1
    print(f"\n✅ {label} done. Missing files: {len(missing_files)}")
    if alt_found:
        print("\nAlternative files used:")
        for orig, alt in alt_found:
            print(f"{orig} --> {alt}")

# Read CSVs
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


print("\n🔍 Processing TRAIN split...")
process_split(train_df, OUTPUT_TRAIN, label="Train")

print("\n🔍 Processing TEST split...")
process_split(test_df, OUTPUT_TEST, label="Test")
