import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os

# -----------------------------
# 1. SETTINGS
# -----------------------------
DATASET_PATH = "dataset"
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. IMAGE TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# 3. LOAD DATASET
# -----------------------------
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# -----------------------------
# 4. MODEL (TRANSFER LEARNING)
# -----------------------------
model = models.resnet18(pretrained=True)

# Freeze early layers (faster training)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

model = model.to(device)

# -----------------------------
# 5. LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# -----------------------------
# 6. TRAINING LOOP
# -----------------------------
print("\nTraining started...\n")

for epoch in range(EPOCHS):
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")

# -----------------------------
# 7. SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "human_classifier.pth")
print("\nModel saved as human_classifier.pth")

# -----------------------------
# 8. LOAD MODEL FOR INFERENCE
# -----------------------------
model.eval()

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    return dataset.classes[predicted.item()]

# -----------------------------
# 9. TEST EXAMPLE
# -----------------------------
if __name__ == "__main__":
    test_image = "test.jpg"  # change this

    if os.path.exists(test_image):
        result = predict(test_image)
        print("\nPrediction:", result)
    else:
        print("\nPut a test.jpg image in the folder to test prediction.")
