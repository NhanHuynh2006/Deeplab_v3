import torch
from torch.utils.data import DataLoader
from dataset import RoadDataset

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    input_size=(240, 240),
    num_workers=0,
    pin_memory=False
):
    """
    Tạo dataloaders cho training và validation
    """
    # Dataset cho training (có data augmentation)
    train_ds = RoadDataset(
        image_path=train_dir,
        mask_path=train_maskdir,
        size=input_size,
        augment=True
    )
    
    # Dataset cho validation (không data augmentation)
    val_ds = RoadDataset(
        image_path=val_dir,
        mask_path=val_maskdir,
        size=input_size,
        augment=False
    )
    
    # DataLoader cho training
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # DataLoader cho validation
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda", num_classes=2):
    """
    Đánh giá mô hình segmentation đa lớp
    """
    model.eval()
    correct = 0
    total_pixels = 0
    total_loss = 0.0
    dice_scores = []
    intersection_total = torch.zeros(num_classes).to(device)
    union_total = torch.zeros(num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # shape: (B, H, W)

            logits = model(x)  # shape: (B, C, H, W)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)  # shape: (B, H, W)
            correct += (preds == y).sum().item()
            total_pixels += torch.numel(y)

            for cls in range(num_classes):
                pred_cls = (preds == cls)
                true_cls = (y == cls)
                intersection = (pred_cls & true_cls).sum()
                union = (pred_cls | true_cls).sum()
                intersection_total[cls] += intersection
                union_total[cls] += union

                # Dice score cho mỗi class
                dice = (2. * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)
                dice_scores.append(dice.item())

    # Tính các metric
    acc = 100 * correct / total_pixels
    miou = (intersection_total / (union_total + 1e-8)).mean().item()
    mean_dice = sum(dice_scores) / len(dice_scores)
    avg_loss = total_loss / len(loader)

    # In kết quả
    print(f"Accuracy: {acc:.2f}%")
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"mIoU: {miou:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # In thêm IoU cho từng class
    for cls in range(num_classes):
        iou = intersection_total[cls] / (union_total[cls] + 1e-8)
        print(f"Class {cls} IoU: {iou.item():.4f}")

    model.train()
    return acc, mean_dice, miou


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, device=None):
    """
    Load checkpoint và trả về epoch hiện tại
    """
    if device is None:
        device = next(model.parameters()).device
    
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Đảm bảo state của optimizer cũng ở trên cùng device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    return checkpoint["epoch"]
