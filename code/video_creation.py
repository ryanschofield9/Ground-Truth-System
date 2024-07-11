import cv2
import os

image_folder = 'video_images/W1B1/full run'
video_name = 'video_images/W1B1/full run/tree_fullrun.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for num in range (0, len(images)):
    image = 'frame' +str(num) +'.png'
    print(image)
    video.write(cv2.imread(os.path.join(image_folder, image)))
    

cv2.destroyAllWindows()
video.release()