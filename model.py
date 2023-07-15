import torch
import torch.nn as nn
import torchvision.models as models

# 载入预训练的ResNet34模型
resnet = models.resnet34(pretrained=True)

# 冻结所有模型参数
for param in resnet.parameters():
    param.requires_grad = False

# 替换最后的全连接层，适应于1778类的车型识别任务
num_classes = 1778
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到设备
resnet = resnet.to(device)

# 输出模型结构
print(resnet)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)


# 示例训练过程
def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        # 训练模式
        model.train()

        # 训练过程...
        # 前向传播、计算损失、反向传播、优化模型参数等

        # 示例：
        # for inputs, labels in train_loader:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #
        #     optimizer.zero_grad()
        #
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #
        #     loss.backward()
        #     optimizer.step()

        # 打印训练过程中的损失等信息
        # print("Epoch: {} Loss: {}".format(epoch+1, loss.item()))

        # 验证模式
        model.eval()

        # 验证过程...
        # 示例：
        # total_correct = 0
        # total_samples = 0
        #
        # with torch.no_grad():
        #     for inputs, labels in val_loader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #
        #         outputs = model(inputs)
        #         _, predicted = torch.max(outputs, 1)
        #
        #         total_samples += labels.size(0)
        #         total_correct += (predicted == labels).sum().item()
        #
        #     accuracy = total_correct / total_samples
        #
        #     print("Validation Accuracy: {}".format(accuracy))

# 示例调用
# train_model(resnet, criterion, optimizer, num_epochs=10)
