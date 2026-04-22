import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================
# CONFIG
# =========================
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda")
BATCH_SIZE = 16
EPOCHS = 8
DATA_ROOT = "C:/Sampath"
MODEL_SAVE_PATH = "best_model.pth"

print("Using device:", DEVICE)

# =========================
# DATASET
# =========================
class CTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        classes = ["NTM", "TB"]

        for label, cls in enumerate(classes):
            class_path = os.path.join(root_dir, cls)
            if not os.path.exists(class_path):
                continue

            for case in os.listdir(class_path):
                case_path = os.path.join(class_path, case)
                if not os.path.isdir(case_path):
                    continue

                for file in os.listdir(case_path):
                    if file.endswith(".png"):
                        img_path = os.path.join(case_path, file)
                        self.samples.append((img_path, label, case))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, patient_id = self.samples[idx]
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32), patient_id


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

train_dataset = CTDataset(os.path.join(DATA_ROOT, "train"), transform)
val_dataset = CTDataset(os.path.join(DATA_ROOT, "val"), transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0, pin_memory=True)

# =========================
# MODEL
# =========================
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scaler = torch.amp.GradScaler("cuda")

best_auc = 0

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for images, labels, _ in tqdm(train_loader):

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.unsqueeze(1).to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    print(f"\nEpoch {epoch+1} Train Loss: {running_loss/len(train_loader):.4f}")

    # =========================
    # VALIDATION (PATIENT LEVEL)
    # =========================
    model.eval()
    patient_probs = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for images, labels, patient_ids in val_loader:

            images = images.to(DEVICE, non_blocking=True)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            for prob, label, pid in zip(probs, labels, patient_ids):
                patient_probs[pid].append(prob[0])
                patient_labels[pid] = label.item()

    final_preds = []
    final_labels = []
    final_probs = []

    for pid in patient_probs:
        mean_prob = sum(patient_probs[pid]) / len(patient_probs[pid])
        pred = 1 if mean_prob > 0.5 else 0

        final_preds.append(pred)
        final_labels.append(int(patient_labels[pid]))
        final_probs.append(mean_prob)

    acc = accuracy_score(final_labels, final_preds)
    auc = roc_auc_score(final_labels, final_probs)

    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved!")

print("\nTraining complete.")