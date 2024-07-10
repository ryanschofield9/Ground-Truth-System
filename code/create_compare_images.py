import cv2 
import matplotlib.pyplot as plt
endings = [".png", "scoring_white1_black1.png", "scoring_white1_black05.png", "scoring_white2_black1.png"]
titles = ["40%", "W1B1", "W1B0.5", "W2B1"]
for frame in range (70, 80):
    fig=plt.figure(frame)
    for spot in range (0,4):
        filename = "video_images/Just Final Image/frame" + str(frame) + endings[spot]
        img =  cv2.imread(filename)
        fig.add_subplot(2,2,spot+1)
        plt.imshow(img, cmap='gray')
    filename = 'video_images/Just Final Image/'+ 'frame' + str(frame) + 'all.png'
    plt.savefig(filename)
    #plt.show()