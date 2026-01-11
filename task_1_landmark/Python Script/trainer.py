import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import FetalLandmarkDataset
from model import SimpleUNet

# --- CONFIG ---
CSV_PATH = '../../role_challenge_dataset_ground_truth.csv' # Adjust path relative to folder
IMAGES_DIR = '../../images' 
EPOCHS = 50
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    # Load Data
    full_ds = FetalLandmarkDataset(CSV_PATH, IMAGES_DIR)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = SimpleUNet(in_channels=1, out_channels=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Starting training on {DEVICE}...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for imgs, hms in train_loader:
            imgs, hms = imgs.to(DEVICE), hms.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, hms) * 1000.0 # Scaling factor
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, hms in val_loader:
                imgs, hms = imgs.to(DEVICE), hms.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, hms).item() * 1000.0
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), '../Model Weights/best_model_landmark.pth')
            print("Saved Best Model.")

if __name__ == "__main__":
    train()