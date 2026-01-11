import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleUNet
from dataset import FetalLandmarkDataset

# --- CONFIG ---
WEIGHTS_PATH = '../Model Weights/best_model_landmark.pth'
CSV_PATH = '../../role_challenge_dataset_ground_truth.csv'
IMAGES_DIR = '../../images'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_coords(heatmap):
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return x, y

def visualize():
    # Load Dataset (use a few samples)
    ds = FetalLandmarkDataset(CSV_PATH, IMAGES_DIR)
    
    # Load Model
    model = SimpleUNet(1, 4).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    
    indices = [0, 1, 2] # Test first 3 valid images
    
    plt.figure(figsize=(15, 5))
    
    for i, idx in enumerate(indices):
        img_tensor, gt_hm = ds[idx]
        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_hm = model(input_tensor).squeeze().cpu().numpy()
            
        img = img_tensor.squeeze().numpy()
        
        plt.subplot(1, 3, i+1)
        plt.imshow(img, cmap='gray')
        
        # Plot Prediction
        points = [get_coords(pred_hm[k]) for k in range(4)]
        plt.scatter([p[0] for p in points], [p[1] for p in points], c='red', marker='x', s=50, label='Pred')
        
        # Draw Lines
        plt.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], 'r--', alpha=0.7)
        plt.plot([points[2][0], points[3][0]], [points[2][1], points[3][1]], 'r--', alpha=0.7)
        
        plt.title(f"Sample {idx}")
        if i==0: plt.legend()
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()