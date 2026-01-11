import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FetalSegmentationDataset
from model import SimpleUNet

CSV_PATH = '../../role_challenge_dataset_ground_truth.csv'
IMAGES_DIR = '../../images'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    ds = FetalSegmentationDataset(CSV_PATH, IMAGES_DIR)
    train_loader = DataLoader(ds, batch_size=8, shuffle=True)
    
    model = SimpleUNet(1, 1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Segmentation Training...")
    
    for epoch in range(50):
        model.train()
        total_loss = 0
        for img, mask in train_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")
        
    torch.save(model.state_dict(), '../Model Weights/best_model_segmentation.pth')
    print("Model Saved.")

if __name__ == "__main__":
    train()