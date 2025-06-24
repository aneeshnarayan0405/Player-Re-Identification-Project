import cv2
from ultralytics import YOLO

model = YOLO('models/best.pt')

def extract_detections(video_path, label, sample_rate=10):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    player_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % sample_rate == 0:
            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (64, 128))
                        feature = crop_resized.flatten() / 255.0
                        player_features.append({
                            'frame': frame_id,
                            'label': label,
                            'box': (x1, y1, x2, y2),
                            'feature': feature
                        })
        frame_id += 1
    cap.release()
    return player_features
