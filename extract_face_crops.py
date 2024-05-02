import json
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path
import cv2
from tqdm import tqdm

# This code reads the bounding boxes for each frame and crop it to have only the faces

def extract_faces(video_path, video_id, bbox_file, output_dir):
    # Read bounding boxes from JSON file
    with open(bbox_file, 'r') as file:
        bboxes = json.load(file)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output directory for this video
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Process each frame
    for frame_index in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if there are bounding boxe for the current frame
        if frame_index < len(bboxes):
            for idx, bbox in enumerate(bboxes[frame_index]):
                #x, y, w, h = [int(b) for b in bbox]
                #crop_img = frame[y:y+h, x:x+w]
                xmin, ymin, xmax, ymax = [int(b) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = h // 3
                p_w = w // 3
                crop_img = frame[max(ymin - p_h, 0):ymax+p_h, max(xmin - p_w, 0):xmax + p_w]
                crop_filename = f"{video_id}_{frame_index}.png"
                cv2.imwrite(os.path.join(video_output_dir, crop_filename), crop_img)

    cap.release()

def process_videos(video_dir, boxes_dir, output_dir):
    # Load the video paths and their IDs from the CSV file
    dataset = pd.read_csv(os.path.join(os.path.dirname(video_dir), 'experimental_dataset.csv'))
    tasks = []

    for index, row in dataset.iterrows():
        video_filename = row['file']
        video_id = str(row['id'])
        video_path = os.path.join(video_dir, video_filename)
        bbox_file = os.path.join(boxes_dir, f"{video_id}.json")
        
        if os.path.exists(bbox_file):
            tasks.append((video_path, video_id, bbox_file, output_dir))
    
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_video, tasks), total=len(tasks)))

def process_video(args):
    video_path, video_id, bbox_file, output_dir = args
    extract_faces(video_path, video_id, bbox_file, output_dir)

if __name__ == "__main__":
    video_dir = r'C:/Users/Public/Hackatons/Deepfake-Detection/dataset/experimental_dataset'
    output_dir = os.path.join(video_dir, 'crops')
    boxes_dir = os.path.join(video_dir, 'boxes')

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_videos(video_dir, boxes_dir, output_dir)
