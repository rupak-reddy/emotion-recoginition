import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_classes = 7  # this is adjustable by me. I dont know the exact details


transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

data_dir = "dataset" # path should be adjusted. I havent wrote any because i dont have exact details
datasets = {
    "train": datasets.ImageFolder(root=f"{data_dir}/train", transform=transform["train"]),
    "val": datasets.ImageFolder(root=f"{data_dir}/val", transform=transform["val"]),
}

dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
    "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
}


model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  


model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


def train_and_validate():
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                
                optimizer.zero_grad()

                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double() / len(datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            
            if phase == "val" and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                torch.save(model.state_dict(), "best_model.pth")

    print(f"Best Validation Accuracy: {best_accuracy:.4f}")


def evaluate_model():
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=datasets["val"].classes))


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=datasets["val"].classes, yticklabels=datasets["val"].classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


train_and_validate()
evaluate_model()


