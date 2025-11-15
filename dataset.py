import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as ff
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class LabelProcessor:
    def __init__(self):
        # 1 background + 1 road class
        self.colormap = [
            (0, 0, 0),         # background
            (128, 64, 128)     # road
        ]
        # Dùng dict thay vì mảng lớn
        self.color2label = self.encode_label_pix(self.colormap)

    def encode_label_pix(self, colormap):
        return {tuple(color): idx for idx, color in enumerate(colormap)}

    def encode_label_img(self, img):
        """Chuyển ảnh RGB sang chỉ số class. Nếu ảnh đã là index, thì giữ nguyên."""
        if img.mode == 'L':  # Nếu ảnh đã là index (chỉ số lớp)
            return np.array(img, dtype='int64')
        else:
            data = np.array(img, dtype='int32')
            label = np.zeros((data.shape[0], data.shape[1]), dtype='int64')
            for rgb, idx in self.color2label.items():
                mask = np.all(data == rgb, axis=-1)  # Kiểm tra các pixel RGB
                label[mask] = idx
            return label

# Khởi tạo label processor
label_processor = LabelProcessor()

class RoadDataset(Dataset):
    def __init__(self, image_path, mask_path, size=(240, 240), augment=True):
        self.image_path = image_path
        self.mask_path = mask_path
        self.size = size
        self.augment = augment
        
        # Lọc file ảnh hợp lệ
        valid_exts = [".png", ".jpg", ".jpeg"]
        self.image_files = sorted([f for f in os.listdir(image_path) 
                                  if os.path.splitext(f)[1].lower() in valid_exts])
        self.mask_files = sorted([f for f in os.listdir(mask_path) 
                                 if os.path.splitext(f)[1].lower() in valid_exts])
        
        print(f"Found {len(self.image_files)} images in {image_path}")
        print(f"Found {len(self.mask_files)} masks in {mask_path}")
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks count!"
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Đọc ảnh và mask
        img_path = os.path.join(self.image_path, self.image_files[idx])
        mask_path = os.path.join(self.mask_path, self.mask_files[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        
        # Resize đầu vào
        img, mask = self.resize(img, mask, self.size)
        
        # Áp dụng transform
        if self.augment:
            img, mask = self.transform_with_augmentation(img, mask)
        else:
            img, mask = self.transform_without_augmentation(img, mask)
            
        return img, mask
    
    def resize(self, img, mask, size):
        img = ff.resize(img, size=size, interpolation=InterpolationMode.BILINEAR)
        mask = ff.resize(mask, size=size, interpolation=InterpolationMode.NEAREST)
        return img, mask
    
    def transform_with_augmentation(self, img, mask):
        # Tạo seed để áp dụng transform đồng bộ cho img và mask
        seed = np.random.randint(2147483647)
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = ff.hflip(img)
            mask = ff.hflip(mask)
        
        # Random rotation
        if np.random.random() > 0.5:
            angle = transforms.RandomRotation.get_params((-10, 10))
            img = ff.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = ff.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
            
        # Color jitter cho img (không áp dụng cho mask)
        jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                      saturation=0.2, hue=0.1)
        img = jitter(img)
        
        # Chuyển thành tensor
        img = transforms.ToTensor()(img)
        
        # Chuẩn hóa img
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        
        # Chuyển mask sang tensor
        mask = label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask).long()
        
        return img, mask
    
    def transform_without_augmentation(self, img, mask):
        # Chuyển thành tensor
        img = transforms.ToTensor()(img)
        
        # Chuẩn hóa img
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        
        # Chuyển mask sang tensor
        mask = label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask).long()
        
        return img, mask

# Test
if __name__ == "__main__":
    # Đường dẫn test
    image_dir = "/home/nhan-huynh/Documents/data/ảnh lọcV1"
    mask_dir = "/home/nhan-huynh/Documents/data/mask2"
    
    # Kiểm tra nếu thư mục tồn tại
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        dataset = RoadDataset(image_dir, mask_dir)
        print(f"Dataset size: {len(dataset)}")
        
        # Lấy một mẫu
        img, mask = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
    else:
        print(f"Thư mục không tồn tại, hãy cập nhật đường dẫn phù hợp")
