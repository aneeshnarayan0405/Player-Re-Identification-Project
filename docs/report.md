# Player Re-Identification Report

## Approach

We implemented two tasks for player re-identification using a YOLOv11-based detection model.

---

## Option 1: Cross-Camera Player Mapping

### Methodology:
- Detected players in both `broadcast.mp4` and `tacticam.mp4` using YOLOv11.
- Extracted visual features by resizing and flattening player crops.
- Used cosine similarity to match player features across both feeds.
- Assigned consistent player IDs and saved annotated videos.

### Techniques Tried:
- Pixel-based embedding (baseline)
- Considered CNN-based embedding for future improvement

---

## Option 2: Re-ID in Single Feed

### Methodology:
- Detected players in each frame of `15sec_input_720p.mp4`.
- Tracked visual features over time using cosine similarity.
- Re-assigned same ID to players who exited and re-entered the frame.

### Visualization:
- Tracked ID distribution over time
- Sample frames shown with bounding boxes and ID tags

---

## Challenges
- Variability in pose and scale between cameras
- Loss of identity in overlapping frames (can be solved with DeepSORT or optical flow)

## Improvements
- Add embedding from CNN/ResNet
- Implement DeepSORT for real-time improvements

## Status
- ✅ Complete and tested
- 🧪 Results saved to `/results`
