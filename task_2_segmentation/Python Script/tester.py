import torch
import cv2
import matplotlib.pyplot as plt
from model import SimpleUNet
from dataset import FetalSegmentationDataset

WEIGHTS = '../Model Weights/best_model_segmentation.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    ds = FetalSegmentationDataset('../../role_challenge_dataset_ground_truth.csv', '../../images')
    model = SimpleUNet(1, 1).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()
    
    img, mask = ds[0]
    with torch.no_grad():
        pred = torch.sigmoid(model(img.unsqueeze(0).to(DEVICE)))
        
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(img.squeeze(), cmap='gray'); plt.title("Input")
    plt.subplot(1, 2, 2); plt.imshow(pred.squeeze().cpu() > 0.5, cmap='gray'); plt.title("Predicted Mask")
    plt.show()

if __name__ == "__main__":
    test()