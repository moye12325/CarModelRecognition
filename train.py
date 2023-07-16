import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# 构建ResNet-34模型
def build_resnet34(num_classes):
    model = models.resnet34(pretrained=True)

    # 冻结预训练的层参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换最后一层全连接层
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

# 定义数据集路径
data_dir = "./data"  # 替换为车型数据集的路径

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
dataset = ImageFolder(data_dir, transform=transform)

# 定义交叉验证的折数
num_folds = 5

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    model.to(device)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)
            train_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct.double() / len(train_loader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * images.size(0)
                val_correct += torch.sum(preds == labels.data)

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct.double() / len(val_loader.dataset)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 进行K折交叉验证
kf = KFold(n_splits=num_folds, shuffle=True)
fold = 1

for train_index, val_index in kf.split(dataset):
    print(f"Fold: {fold}")
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 构建模型
    num_classes = len(dataset.classes)
    model = build_resnet34(num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # 保存模型（可根据需要保存每个fold的模型）
    torch.save(model.state_dict(), f"resnet34_model_fold{fold}.pth")

    fold += 1
