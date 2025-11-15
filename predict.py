import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from model import create_deeplabv3

def predict_image(model, image_path, device="cuda", size=(240, 240)):
    """
    Dự đoán segmentation mask cho một ảnh
    """
    # Đưa model sang eval mode
    model.eval()
    
    # Đọc ảnh
    image = Image.open(image_path).convert("RGB")
    
    # Resize
    image_resized = image.resize(size, Image.BILINEAR)
    
    # Chuyển sang tensor và chuẩn hóa
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0).to(device)
    
    # Dự đoán
    with torch.no_grad():
        output = model(input_tensor)
        
    # Lấy class có xác suất cao nhất
    pred = torch.argmax(output, dim=1)
    pred = pred.cpu().numpy()[0]
    
    return pred, image

def colorize_mask(mask):
    """
    Chuyển mask thành ảnh màu để visualize
    """
    # Tạo ảnh trống
    colorized = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Áp dụng màu cho mỗi class
    # Class 0: Background - Đen
    # Class 1: Road - Tím
    colors = [
        [0, 0, 0],         # Background - Đen
        [128, 64, 128],    # Road - Tím
    ]
    
    for class_idx, color in enumerate(colors):
        colorized[mask == class_idx] = color
    
    return colorized

def overlay_mask(image, mask, alpha=0.5):
    """
    Chồng segmentation mask lên ảnh gốc
    """
    # Convert PIL image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Colorize mask
    colored_mask = colorize_mask(mask)
    
    # Resize mask to match image dimensions if needed
    if image.shape[:2] != mask.shape[:2]:
        colored_mask = np.array(Image.fromarray(colored_mask).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST
        ))
    
    # Blend images
    blended = image * (1 - alpha) + colored_mask * alpha
    
    return blended.astype(np.uint8)

def main():
    # Đường dẫn đến checkpoint
    checkpoint_path = "/home/nhan-huynh/Downloads/Deeplearning/Deeplearning_V1 (Copy)/deeplab/checkpoints/checkpoint_best.pth.tar"
    
    # Thư mục chứa ảnh test
    test_dir = "/home/nhan-huynh/Documents/data/ảnh lọcV1"
    
    # Thư mục lưu kết quả
    output_dir = "/home/nhan-huynh/Downloads/Deeplearning/Deeplearning_V1 (Copy)/deeplab/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tạo model
    model = create_deeplabv3(num_classes=2)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch')}")
        print(f"Metrics - mIoU: {checkpoint.get('miou', 'N/A')}, "
              f"Dice: {checkpoint.get('dice', 'N/A')}, "
              f"Acc: {checkpoint.get('acc', 'N/A')}%")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return
    
    # Đưa model lên device
    model = model.to(device)
    
    # Đọc danh sách ảnh test
    valid_exts = [".png", ".jpg", ".jpeg"]
    test_images = [f for f in os.listdir(test_dir) 
                  if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(test_images)} test images")
    
    # Dự đoán và lưu kết quả
    for idx, img_name in enumerate(test_images):
        if idx >= 10:  # Chỉ test 10 ảnh đầu tiên
            break
        
        img_path = os.path.join(test_dir, img_name)
        
        try:
            # Dự đoán
            mask, original_image = predict_image(model, img_path, device)
            
            # Tạo overlay
            overlay = overlay_mask(original_image, mask, alpha=0.5)
            
            # Tạo figure
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_image)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(colorize_mask(mask))
            plt.title("Segmentation Mask")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title("Overlay")
            plt.axis("off")
            
            # Lưu kết quả
            output_path = os.path.join(output_dir, f"result_{img_name}")
            plt.savefig(output_path)
            plt.close()
            
            print(f"Processed and saved: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
