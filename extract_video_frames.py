import cv2
import numpy as np

def extract_frames(video_path, num_frames=30, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(frame_count // num_frames, 1)
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    
    cap.release()
    
    # Convert list of frames to numpy array with shape (num_frames, height, width, channels)
    frames = np.array(frames)

    return frames

# Example usage:
video_path = "short_video.mp4"  # Path to the downloaded video
video_frames = extract_frames(video_path)
print(video_frames.shape)  # Should be (30, 128, 128, 3)
