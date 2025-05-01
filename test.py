# ============================================================
# Food-91 classification – clean, reproducible implementation
# ============================================================

# ---------- 1. Imports ----------
import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# ---------- 2. Reproducibility ----------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # deterministic GPU kernels
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- 3. Dataset ----------
class FoodDataset(Dataset):
    """
    Loads images from subfolders. Folder name = class label (string).
    """
    def __init__(self, root_dir, encoder, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.encoder = encoder
        self.paths, str_labels = [], []

        for folder in os.scandir(root_dir):
            if not folder.is_dir():          # skip stray files
                continue
            for img in os.scandir(folder.path):
                self.paths.append(img.path)
                str_labels.append(folder.name)

        self.labels = self.encoder.transform(str_labels)

    def __len__(self):                  # length of dataset
        return len(self.paths)

    def __getitem__(self, idx):         # returns (image, label) pair
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

    # helper to compute mean / std for normalisation
    def get_mean_std(self, sample_size=5000):
        idxs = np.random.choice(len(self.paths), size=min(sample_size, len(self.paths)), replace=False)
        mean = np.zeros(3); sq_mean = np.zeros(3); n = 0
        for i in idxs:
            arr = np.array(Image.open(self.paths[i]).convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
            mean += arr.mean(axis=(0, 1))
            sq_mean += (arr ** 2).mean(axis=(0, 1))
            n += 1
        mean /= n
        std = np.sqrt(sq_mean / n - mean ** 2)
        return mean.tolist(), std.tolist()

# ---------- 4. Label Encoder ----------
class_names = [f.name for f in os.scandir("train") if f.is_dir()]
encoder = LabelEncoder().fit(class_names)
num_classes = len(class_names)

# ---------- 5. Build datasets ----------
print("computing mean/std on train set …")
train_tmp = FoodDataset("train", encoder)         # temp without transform
mean_rgb, std_rgb = train_tmp.get_mean_std()
print("train mean:", mean_rgb, "  train std:", std_rgb)

train_tfms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean_rgb, std_rgb)                 # same stats for all splits
])
test_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean_rgb, std_rgb)
])

train_ds = FoodDataset("train", encoder, transform=train_tfms)
test_ds  = FoodDataset("test" , encoder, transform=test_tfms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds , batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# ---------- 6. CNN model ----------
class FoodCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                            # 64 × 112 × 112

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                            # 128 × 56 × 56

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),                            # 256 × 28 × 28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)             # 256 × 1 × 1
        self.classifier = nn.Linear(256, num_classes)    # raw logits

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

# ---------- 7. Helper functions ----------
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item() * 100

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_corr, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_corr += accuracy(logits, y) * x.size(0)
        n += x.size(0)
    return total_corr / n

# ---------- 8. Training loop ----------
model = FoodCNN(num_classes).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

best_acc = 0.0
for epoch in range(1, 21):
    model.train()
    epoch_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    test_acc = evaluate(model, test_loader)
    print(f"Epoch {epoch:02d} | loss {epoch_loss:.3f} | test acc {test_acc:.2f}%")

    if test_acc > best_acc:
        torch.save(model.state_dict(), "best_model.pth")
        best_acc = test_acc
    scheduler.step()

# ---------- 9. Final test accuracy ----------
model.load_state_dict(torch.load("best_model.pth"))
final_acc = evaluate(model, test_loader)
print(f"\nFinal Test Accuracy: {final_acc:.2f}%")

# ---------- 10. Hyper-parameters summary ----------
print("\nHyper-parameters used:")
print(f"- batch_size     : 16")
print(f"- learning_rate  : 1e-3 (Adam), StepLR gamma=0.1 @ epoch 8")
print(f"- weight_decay   : 1e-4")
print(f"- epochs         : 20")
print(f"- image size     : 224×224")
print(f"- normalisation  : mean={mean_rgb}, std={std_rgb}")
print(f"- random seed    : 42  (deterministic cuDNN)")