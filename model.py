import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet50', pretrained=True, output_stride=8):
        super(DeepLabV3, self).__init__()
        
        # Lựa chọn backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'mobilenetv2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained).features
            return_layers = {'18': 'out'}
            self.backbone = models._utils.IntermediateLayerGetter(self.backbone, return_layers)
            in_channels = 320
        else:
            raise ValueError(f'Backbone {backbone} không được hỗ trợ')
        
        # Đối với ResNet, chúng ta sử dụng layer4 là lớp cuối cùng
        if backbone in ['resnet50', 'resnet101']:
            return_layers = {'layer4': 'out'}
            self.backbone = models._utils.IntermediateLayerGetter(self.backbone, return_layers)
            in_channels = 2048
        
        # Thêm DeepLabHead đặc trưng (ASPP)
        self.classifier = DeepLabHead(in_channels, num_classes)
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features["out"])
        
        # Upsample lại theo kích thước ảnh gốc
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        
        return x


def create_deeplabv3(num_classes=2, backbone='resnet50', pretrained=True, output_stride=8):
    """
    Hàm helper để tạo model DeepLabV3 với tham số tùy chỉnh
    """
    model = DeepLabV3(num_classes=num_classes, backbone=backbone, 
                     pretrained=pretrained, output_stride=output_stride)
    return model


def test():
    # Test với ảnh 3x240x240
    x = torch.randn((2, 3, 240, 240))
    model = create_deeplabv3(num_classes=2)
    preds = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == (2, 2, 240, 240), f"Expected shape (2, 2, 240, 240), got {preds.shape}"
    
if __name__ == "__main__":
    test()
