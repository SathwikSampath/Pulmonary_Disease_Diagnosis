import os
import random
import shutil

# ========================
# SETTINGS
# ========================
random.seed(42)

SOURCE_DIR = "C:/Sampath/Segmented"
DEST_DIR = "C:/Sampath"

SPLIT_RATIO = (0.7, 0.15, 0.15)

classes = ["NTM", "TB"]

# ========================
# CREATE DEST FOLDERS
# ========================
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)

# ========================
# SPLIT FUNCTION
# ========================
def split_patients(class_name):
    class_path = os.path.join(SOURCE_DIR, class_name)
    patients = os.listdir(class_path)
    
    random.shuffle(patients)
    
    total = len(patients)
    train_end = int(SPLIT_RATIO[0] * total)
    val_end = train_end + int(SPLIT_RATIO[1] * total)
    
    train_patients = patients[:train_end]
    val_patients = patients[train_end:val_end]
    test_patients = patients[val_end:]
    
    return train_patients, val_patients, test_patients

# ========================
# EXECUTE SPLIT
# ========================
for cls in classes:
    train_p, val_p, test_p = split_patients(cls)
    
    for patient in train_p:
        shutil.copytree(
            os.path.join(SOURCE_DIR, cls, patient),
            os.path.join(DEST_DIR, "train", cls, patient)
        )
        
    for patient in val_p:
        shutil.copytree(
            os.path.join(SOURCE_DIR, cls, patient),
            os.path.join(DEST_DIR, "val", cls, patient)
        )
        
    for patient in test_p:
        shutil.copytree(
            os.path.join(SOURCE_DIR, cls, patient),
            os.path.join(DEST_DIR, "test", cls, patient)
        )

print("Patient-level split completed successfully.")