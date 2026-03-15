
import torch
import timm
import torch.nn as nn

def check_model():
    model_name = "resnet50"
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
    backbone.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 320, 320)
    with torch.no_grad():
        output = backbone(dummy_input)
    
    print(f"Backbone output shape: {output.shape}")
    print(f"Backbone output range: {output.min().item():.4f} to {output.max().item():.4f}")
    
    in_features = output.shape[-1]
    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 2),
    )
    
    # Test head
    head.eval() # Important for BN
    with torch.no_grad():
        logits = head(output)
    
    print(f"Head output shape: {logits.shape}")
    print(f"Head output range: {logits.min().item():.4f} to {logits.max().item():.4f}")

if __name__ == "__main__":
    check_model()
