import os
import timm
import torch

def download_pretrained_weights():
    """
    Download and save pretrained weights locally.
    """
    print("Downloading pretrained weights for tf_efficientnetv2_b2...")
    
    # Create weights directory if it doesn't exist
    weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download and save weights
    model = timm.create_model("tf_efficientnetv2_b2", pretrained=True)
    weights_path = os.path.join(weights_dir, "tf_efficientnetv2_b2_pretrained.pth")
    torch.save(model.state_dict(), weights_path)
    
    print(f"Weights saved to {weights_path}")

if __name__ == "__main__":
    download_pretrained_weights()