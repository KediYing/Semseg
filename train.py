import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from PIL import Image
import os
import numpy as np
from torchvision import transforms

# 数据集路径设置
data_dir = 'FoodSeg103/FoodSeg103'
images_dir = os.path.join(data_dir, 'Images', 'img_dir', 'train')
masks_dir = os.path.join(data_dir, 'Images', 'ann_dir', 'train')

from torchvision import transforms

class FoodDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.ids = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.images_dir, img_id)
        mask_id = img_id.replace('.jpg', '.png')  # Assuming mask files are in .png format
        mask_path = os.path.join(self.masks_dir, mask_id)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)  # Convert mask to a tensor
        mask = mask.squeeze()  # Remove the channel dimension if any
        return image, mask

# 定义转换操作，统一图像大小并转换为张量
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整到统一的尺寸，例如256x256
    transforms.ToTensor()
])

# 使用转换
train_dataset = FoodDataset(images_dir, masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置模型
model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
num_classes = 104  # Assume 103 classes + background
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)  # 确保 masks 是正确的维度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")


        # 保存模型的状态字典
        torch.save(model.state_dict(), f'model_state_epoch_{epoch+1}.pth')

# 设置训练周期
num_epochs = 10
train_model(num_epochs)
