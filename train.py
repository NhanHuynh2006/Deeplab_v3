import os
import torch
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model import create_deeplabv3
from utils import get_loaders, check_accuracy, save_checkpoint, load_checkpoint

def train(model, train_loader, val_loader, optimizer, scheduler=None, start_epoch=0, 
          num_epochs=10, device='cuda', checkpoint_dir='checkpoints'):
    """
    Training function với visualization - chỉ lưu latest checkpoint
    """
    # Đảm bảo thư mục checkpoint tồn tại
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Lưu lại lịch sử để vẽ đồ thị
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'miou': [],
        'dice': []
    }
    
    # Chuyển model lên GPU (nếu có)
    model.to(device)
    
    # Bắt đầu training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar với tqdm
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                # Di chuyển data lên GPU
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Cập nhật loss và hiển thị
                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))
                pbar.update(1)
        
        # Tính loss trung bình cho epoch
        epoch_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        # Tính loss trung bình cho validation
        epoch_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        
        # Đánh giá model với các metric
        print("\nModel Evaluation:")
        acc, dice, miou = check_accuracy(val_loader, model, device, num_classes=2)
        
        # Lưu lại các metric
        history['accuracy'].append(acc)
        history['dice'].append(dice)
        history['miou'].append(miou)
        
        # Cập nhật scheduler nếu có
        if scheduler is not None:
            scheduler.step(miou)  # Truyền vào miou để scheduler theo dõi
        
        # Lưu checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'miou': miou,
            'dice': dice,
            'acc': acc
        }
        
        # Chỉ lưu checkpoint mới nhất
        save_checkpoint(checkpoint, filename=os.path.join(checkpoint_dir, f"checkpoint_latest.pth.tar"))
        print(f"=> Saved latest checkpoint at epoch {epoch + 1}")
    
    # Vẽ biểu đồ training và validation loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.plot(history['miou'], label='mIoU')
    plt.plot(history['dice'], label='Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Metrics')
    
    plt.tight_layout()
    
    # Lưu biểu đồ
    metrics_dir = os.path.join(checkpoint_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plt.savefig(os.path.join(metrics_dir, 'training_metrics.png'))
    
    return history

def main():
    # Thư mục data
    train_dir = "/home/nhan-huynh/Downloads/new/images"
    train_maskdir = "/home/nhan-huynh/Downloads/new/mask"
    val_dir = "/home/nhan-huynh/Downloads/new/images"
    val_maskdir = "/home/nhan-huynh/Downloads/new/mask"
    
    # Thư mục checkpoints
    checkpoint_dir = "/home/nhan-huynh/Downloads/Deeplearning/Deeplearning_V1 (Copy)/deeplab/checkpoints"
    
    # Tham số training
    batch_size = 16
    num_epochs = 1000
    learning_rate = 1e-4
    input_size = (240, 240)  # (height, width)
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tạo dataloaders
    train_loader, val_loader = get_loaders(
        train_dir, train_maskdir, val_dir, val_maskdir,
        batch_size=batch_size, input_size=input_size, 
        num_workers=4, pin_memory=True
    )
    
    # Tạo model - DeepLabV3 với ResNet50 backbone
    model = create_deeplabv3(num_classes=2, backbone='resnet50', pretrained=True)
    
    # Chuyển model lên device trước khi tạo optimizer
    model = model.to(device)
    
    # Tối ưu hóa với AdamW
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Tìm checkpoint gần nhất
    start_epoch = 0
    additional_epochs = 500  # Số epoch muốn train thêm
    
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth.tar")
    
    if os.path.exists(checkpoint_path):
        print(f"=> Found latest checkpoint! Loading...")
        # Sửa đổi cách load checkpoint để xử lý vấn đề device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Đảm bảo state của optimizer cũng ở trên cùng device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        loaded_epoch = checkpoint["epoch"]
        
        # Tính số epoch mới
        num_epochs = loaded_epoch + additional_epochs
        start_epoch = loaded_epoch
        print(f"=> Will train for {additional_epochs} more epochs (from {start_epoch} to {num_epochs})")
    
    # Training
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        num_epochs=num_epochs,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    print("Training complete!")

if __name__ == "__main__":
    main()
