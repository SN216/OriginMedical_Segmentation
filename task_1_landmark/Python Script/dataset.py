import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FetalLandmarkDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, mode='train'):
        raw_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.target_size = 256
        
        valid_indices = []
        print(f"[{mode.upper()}] Scanning dataset with 4-Stage Quality Filter...")
        
        for idx, row in raw_data.iterrows():
            img_name = row['image_name']
            img_path = os.path.join(self.root_dir, img_name)
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue
                
            h, w = image.shape
            coords = row[1:].values.astype('float').reshape(4, 2)
            
            # 1. VOID CHECK (Pixel Intensity)
            in_void = False
            for i in range(4):
                x, y = int(coords[i, 0]), int(coords[i, 1])
                if x < 0 or x >= w or y < 0 or y >= h: in_void = True; break
                # taking a 5*5 box across each coordinate to ensure it's not an empty space
                region = image[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
                if np.mean(region) < 15: in_void = True; break
            if in_void: continue

            # 2. INTERSECTION CHECK
            p0, p1 = coords[0], coords[1] # OFD
            p2, p3 = coords[2], coords[3] # BPD
            def ccw(A, B, C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
            if not ((ccw(p0,p2,p3) != ccw(p1,p2,p3)) and (ccw(p0,p1,p2) != ccw(p0,p1,p3))): continue

            # 3. ANGLE CHECK
            vec_ofd = p1 - p0; vec_bpd = p3 - p2
            unit_ofd = vec_ofd / (np.linalg.norm(vec_ofd) + 1e-6)
            unit_bpd = vec_bpd / (np.linalg.norm(vec_bpd) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(np.dot(unit_ofd, unit_bpd), -1, 1)))
            if angle > 90: angle = 180 - angle
            if angle < 45: continue

            # 4. RATIO CHECK (Biological Plausibility)
            ratio = np.linalg.norm(vec_bpd) / (np.linalg.norm(vec_ofd) + 1e-6)
            if ratio < 0.5 or ratio > 1.1: continue

            valid_indices.append(idx)
                
        self.data = raw_data.iloc[valid_indices].reset_index(drop=True)
        print(f"[{mode.upper()}] Cleaned Data: {len(self.data)} images kept.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        h_orig, w_orig = image.shape
        coords = self.data.iloc[idx, 1:].values.astype('float32').reshape(4, 2)
        
        # Resize
        img_resized = cv2.resize(image, (self.target_size, self.target_size))
        scale_x = self.target_size / w_orig
        scale_y = self.target_size / h_orig
        coords[:, 0] *= scale_x
        coords[:, 1] *= scale_y

        # Heatmaps (Sigma=6 for clear targets)
        heatmaps = self._generate_heatmaps(coords, (self.target_size, self.target_size), sigma=6)
        
        img_tensor = torch.from_numpy(img_resized).float() / 255.0
        return img_tensor.unsqueeze(0), torch.from_numpy(heatmaps).float()

    def _generate_heatmaps(self, landmarks, img_shape, sigma):
        h, w = img_shape
        heatmaps = np.zeros((4, h, w), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        for i in range(4):
            x, y = landmarks[i]
            dist = (xx - x)**2 + (yy - y)**2
            heatmaps[i] = np.exp(-dist / (2 * sigma**2))
        return heatmaps