import cv2
import json
import os
import pandas as pd
from facenet_pytorch import MTCNN
import torch

# This code saves bounding boxes of the face in videos
# Each video has a list of lists, and each frame has a list
# The four elements of each list are:
# 1. x max of the box
# 2. y max of the box
# 3. x min of the box
# 4. y min of the box

def detect_faces_in_video(video_path, detector, device):
    cap = cv2.VideoCapture(video_path)
    frame_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(frame)
        
        if boxes is not None:
            frame_results.append(boxes.tolist())
    
    cap.release()
    return frame_results

def process_videos(directory, output_dir):
    # Load the dataset.csv to map video IDs to filenames
    dataset = pd.read_csv(os.path.join(os.path.dirname(directory), 'experimental_dataset.csv'))[:5]
    
    # Set up the face detector
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=False, min_face_size=100, thresholds=[0.7, 0.8, 0.9], device=device)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each video
    for index, row in dataset.iterrows():
        video_filename = row['file']
        video_path = os.path.join(directory, video_filename)
        print(f"Processing {video_path}")
        
        # Detect faces in the video
        face_data = detect_faces_in_video(video_path, mtcnn, device)
        
        # Save the results
        result_path = os.path.join(output_dir, f"{row['id']}.json")
        with open(result_path, 'w') as outfile:
            json.dump(face_data, outfile)
    
    print("All videos processed.")

if __name__ == "__main__":
    video_dir = r'C:\Users\Public\Hackatons\Deepfake-Detection\dataset\experimental_dataset'
    output_dir = os.path.join(video_dir, 'boxes')
    process_videos(video_dir, output_dir)
