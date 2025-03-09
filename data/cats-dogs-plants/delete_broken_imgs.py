#!/usr/bin/env python3
import os
from PIL import Image

# Update this path to the root directory where your images are stored.
root_dir = "./"

def has_broken_icc_profile(image_path):
    try:
        with Image.open(image_path) as img:
            icc = img.info.get("icc_profile")
            # If an ICC profile exists and its length is not a multiple of 4, return True.
            if icc is not None and (len(icc) % 4 != 0):
                return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    return False

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith(".png"):
            file_path = os.path.join(subdir, file)
            if has_broken_icc_profile(file_path):
                print(f"Deleting file due to broken ICC profile: {file_path}")
                os.remove(file_path)
