import cv2
import os


output_dir = "Frames/"


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


video = cv2.VideoCapture('test3.mp4')


fps = video.get(cv2.CAP_PROP_FPS)


frame_skip = int(fps / 10)


frame_skip = max(1, frame_skip)


frame_count = 0

while True:
    
    ret, frame = video.read()

    
    if not ret:
        break

    
    if frame_count % frame_skip == 0:
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count}.jpg'), frame)

    
    frame_count += 1


video.release()

print(f'Video processed and frames saved as images in {output_dir}.')
