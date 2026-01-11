import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FetalSegmentationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        raw_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        valid_indices = []
        print("Initializing Segmentation Data with Filters...")
        
        for idx, row in raw_data.iterrows():
            img_name = row['image_name']
            img_path = os.path.join(self.root_dir, img_name)
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None: continue
                
            h, w = image.shape
            coords = row[1:].values.astype('float').reshape(4, 2)
            
            # 1. VOID Check
            in_void = False
            for i in range(4):
                x, y = int(coords[i, 0]), int(coords[i, 1])
                if x < 0 or x >= w or y < 0 or y >= h: in_void = True; break
                if np.mean(image[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]) < 15: in_void = True; break
            if in_void: continue

            # 2. Geometry Checks
            p0, p1 = coords[0], coords[1]; p2, p3 = coords[2], coords[3]
            vec_ofd = p1 - p0; vec_bpd = p3 - p2
            ratio = np.linalg.norm(vec_bpd) / (np.linalg.norm(vec_ofd) + 1e-6)
            
            if ratio < 0.5 or ratio > 1.1: continue # Aspect Ratio Filter
            
            valid_indices.append(idx)

        self.data = raw_data.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_name']
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        coords = self.data.iloc[idx, 1:].values.astype('float')
        
        # Generate Ellipse Mask
        mask = np.zeros(image.shape, dtype=np.uint8)
        ofd1, ofd2 = coords[0:2], coords[2:4] # Reshape indices
        bpd1, bpd2 = coords[4:6], coords[6:8]
        
        center = np.mean([ofd1, ofd2], axis=0)
        axes = (np.linalg.norm(ofd1-ofd2)/2, np.linalg.norm(bpd1-bpd2)/2)
        angle = np.degrees(np.arctan2(ofd2[1]-ofd1[1], ofd2[0]-ofd1[0]))
        
        cv2.ellipse(mask, (int(center[0]), int(center[1])), 
                   (int(axes[0]), int(axes[1])), angle, 0, 360, 255, -1)
        
        # Resize
        img = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        return (torch.from_numpy(img).float()/255.0).unsqueeze(0), \
               (torch.from_numpy(mask).float()/255.0).unsqueeze(0)