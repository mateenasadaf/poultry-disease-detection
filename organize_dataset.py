import os
import shutil
import random

# Base dataset path
BASE_DIR = r"C:\Users\mateena Sadaf\Desktop\POULTRY DISEASE\dataset"


# Folders that should NOT be touched
EXISTING_SPLITS = ["train", "test", "valid"]

# Find new disease folders (the ones outside train/test/valid)
disease_folders = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d not in EXISTING_SPLITS
]

print("Detected new disease folders:", disease_folders)

# Create train/test/valid subfolders for each new disease
for disease in disease_folders:
    src_folder = os.path.join(BASE_DIR, disease)

    # create folders
    for split in EXISTING_SPLITS:
        dst_folder = os.path.join(BASE_DIR, split, disease)
        os.makedirs(dst_folder, exist_ok=True)

    # list images
    images = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]

    random.shuffle(images)

    # calculate split sizes
    total = len(images)
    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_imgs = images[:train_end]
    valid_imgs = images[train_end:valid_end]
    test_imgs = images[valid_end:]

    print(f"\n{disease} → {total} images")
    print(f"  Train: {len(train_imgs)}")
    print(f"  Valid: {len(valid_imgs)}")
    print(f"  Test:  {len(test_imgs)}")

    # move images
    for img in train_imgs:
        shutil.move(os.path.join(src_folder, img), os.path.join(BASE_DIR, "train", disease, img))

    for img in valid_imgs:
        shutil.move(os.path.join(src_folder, img), os.path.join(BASE_DIR, "valid", disease, img))

    for img in test_imgs:
        shutil.move(os.path.join(src_folder, img), os.path.join(BASE_DIR, "test", disease, img))

    # remove the empty folder
    os.rmdir(src_folder)

print("\n✔ All new disease folders have been split and moved successfully!")
