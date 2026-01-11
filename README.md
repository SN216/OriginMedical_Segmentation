SN Bibhudutta - Research Intern Challenge Submission

FOLDER STRUCTURE:
- task_1_landmark: Contains the Regression-based heatmap approach (Part A).
- task_2_segmentation: Contains the U-Net Segmentation approach (Part B).

HOW TO RUN:
1. Ensure 'role_challenge_dataset_ground_truth.csv' and 'images' folder are in the root directory relative to the scripts.
2. Navigate to 'Python Script' folder.
3. Run 'python trainer.py' to retrain.
4. Run 'python tester.py' to visualize inference.

RESEARCH HIGHLIGHTS:
- Implemented a 4-Stage Data Quality Filter (dataset.py) to remove broken labels, void-pixels, and geometric outliers.
- Used Heatmap Regression (Part A) instead of coordinate regression for better spatial convergence.
- Used Ellipse Geometry (Part B) to generate ground truth masks from sparse points.
