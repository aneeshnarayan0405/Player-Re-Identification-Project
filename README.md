# Player-Re-Identification-Project

# Player Re-Identification in Sports Footage

## Overview
This project is a submission for the AI Intern assignment at Liat.ai. It includes two comprehensive solutions for player re-identification in sports footage:

- **Option 1: Cross-Camera Player Mapping**
- **Option 2: Re-Identification in a Single Feed**

The goal is to ensure consistent player identification using deep learning and feature tracking techniques. 

## 🔍 Problem Statements

### Option 1: Cross-Camera Player Mapping
Given two clips from different angles (broadcast and tacticam), identify and match each player across views using consistent player IDs.

### Option 2: Re-Identification in a Single Feed
Track players in a single video (`15sec_input_720p.mp4`) and assign the same ID to players re-entering the frame after temporary occlusion or exit.

## 🛠 Technologies & Libraries
- Python 3.10+
- YOLOv11 (Ultralytics)
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

## ✅ Features Implemented

- Real-time object detection using custom-trained YOLOv11
- Cosine similarity based feature matching for player identity
- Modular tracker for re-ID
- Per-frame visual tracking
- Visual analysis plots and annotated video outputs

## 📁 Files Included

- `notebooks/player_reid.ipynb`: Full notebook with code and markdown explanations
- `src/tracker.py`: Custom class for player re-ID tracking
- `src/extract_features.py`: Functions to extract embeddings and detections
- `models/best.pt`: YOLOv11 fine-tuned weights (not included)
- `videos/*.mp4`: Input videos
- `results/`: Output visualizations and video exports
- `docs/report.md`: Methodology, techniques, and challenges summary
- `README.md`: This file

