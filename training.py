import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torch.cuda.amp import autocast, GradScaler

from dataset.HumanLoader import HumanMattingDataset
from model.unet import U_net
from tqdm import tqdm

model = U_net(1).to('cuda')

train_dataset = HumanMattingDataset(split='train')
val_dataset = HumanMattingDataset(split='val')  # Initialize validation dataset

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Initialize validation dataloader

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

num_epochs = 20
best_loss = float('inf')  # Initialize with a high value

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, masks, _, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        images, masks = images.cuda(), masks.cuda()

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print()
            print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Training Loss: {loss.item()}")

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, _, _ in val_loader:
            images, masks = images.cuda(), masks.cuda()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    print()
    print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {val_loss}")

    # Save model if the validation loss is improved
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), r'weights\unet_model_best.pth')
