import cv2
import os

image_folder = 'video_images/W2B1/full run'
video_name = 'video_images/W2B1/full run/zeroangle.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))
frame_num = int(len(frame)/2) 
for num in range (frame_num, frame_num +1):
    image = 'zeroangle.png'
    print(image)
    print(os.path.join(image_folder, image))
    video.write(cv2.imread(os.path.join(image_folder, image)))
    

cv2.destroyAllWindows()
video.release()