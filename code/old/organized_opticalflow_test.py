
import torch
from torchvision.io import read_video
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from torchvision.models.optical_flow import Raft_Small_Weights
from torchvision.models.optical_flow import raft_small
from torchvision.utils import flow_to_image
import copy 
import cv2
from PIL import Image, ImageDraw
import seaborn as sns
import statistics
import io
plt.rcParams["savefig.bbox"] = "tight"
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()

#https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html#sphx-glr-auto-examples-others-plot-optical-flow-py

def plot(imgs,**imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    #plt.show()
    plt.tight_layout()

#https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html#sphx-glr-auto-examples-others-plot-optical-flow-py

def preprocess(img1_batch, img2_batch):
    #img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    #img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)

##https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html#sphx-glr-auto-examples-others-plot-optical-flow-py

def check_occurances(list):
    avail_nums = []
    occurances = []
    paired_occurances = {}
    for num in list: 
        if num not in avail_nums: 
            avail_nums.append(num)
    for avail in avail_nums:
        occurances.append(list.count(avail))
    for (rad, occur) in zip(avail_nums, occurances):
        paired_occurances[rad]=occur
    #https://www.geeksforgeeks.org/sort-python-dictionary-by-value/
    sorted_paired_occurances = dict(sorted(paired_occurances.items(), 
                          key=lambda item: item[1]))
    return sorted_paired_occurances

def create_list (dict): 
    #turn a dictionary with the values (key) and the count of the value (value) into a list with the correct count
    estimates_sorted = []
    for key in dict: 
        times = dict[key]
        for x in range (times):
            estimates_sorted.append(key)
    return estimates_sorted

def create_box (radius, middle, img):
    #draw a box that replicates where the tree is thought to be 
    #calculate left side of the line 
    left_side = int(middle - radius )

    #calculate right side of the line 
    right_side = int (middle + radius) 

    print(f"radius: {radius}, middle: {middle}, left side: {left_side}, right side: {right_side}")

    #draw left side line 
    img = cv2.line(img ,(left_side, 50),(left_side, 400),(155,0,0),5)

    #draw top line 
    img = cv2.line(img ,(left_side, 400),(right_side, 400),(155,0,0),5)

    #draw right side line 
    img = cv2.line(img ,(right_side, 50),(right_side, 400),(155,0,0),5)

    #draw bottom line 
    img = cv2.line(img ,(right_side, 50),(left_side, 50),(155,0,0),5)

    #draw middle line 
    img = cv2.line(img,(int(middle), 0),(int(middle), 480),(155,0,0),5)
#https://www.geeksforgeeks.org/saving-a-plot-as-an-image-in-python/


video_path = "c:\\Users\\ryana\\Documents\\Robotics Research\\video_trial.avi"
frames, _, _ = read_video(str(video_path), output_format="TCHW")
for frame in range (70,80):
    #https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html#sphx-glr-auto-examples-others-plot-optical-flow-py
    #optical flow for the frames in the range above
    batch1 = []
    batch2 = []
    frames_using = frame
    for i in range (frames_using,frames_using+1):
        batch1.append(frames[i])
        batch2.append(frames[i+1])
    img1_batch= torch.stack(batch1)
    img2_batch= torch.stack(batch2)
    img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
    print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = list_of_flows[-1]
    flow_imgs = flow_to_image(predicted_flows)

    # The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
    img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

    grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)] 

    #optical flow image is saved as flowimgs 

    #threshold and filter the flowig to get a black and white image of the tree (black)
    #https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    
    #turn the image into a numpy 
    img2 = flow_imgs[0].numpy()
    # make a copy of the numpy array image 
    flow_saved = copy.deepcopy(img2)
    # save just the numpy array 
    flow_saved = flow_saved[0]
    #move the axis of the image tp be able to normalize 
    img2 = np.moveaxis(img2, 0, -1)

    #normalize image
    img2 = (img2 - min(img2.reshape(-1)))/(max(img2.reshape(-1))-min(img2.reshape(-1)))*255
    img2 = img2.astype(np.uint8)
    print(img2.shape, min(img2.reshape(-1)), max(img2.reshape(-1)))

    #turn the image from RGB to Gray 
    bwimg = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    #Use Gaussian Blur to blur the image to get a better threshold 
    blur = cv2.GaussianBlur(bwimg,(5,5),0)

    #Use OTSU thresholding 
    ret, img=cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #filter the image by opening and closing the image (getting rid of any pixels that are by themselves and filling in any small holes)
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    #set pixel counting variables 
    consecutive = 0 # how many black pixels have I seen in a row 
    total_black = 0 # at the end of a row how many total black pixels have I seen (excluding when there are less than the threshold value)
    black_pixels = [] # a list holding all the black pixel groups over the threshold value that I have seen 
    start = [] # a list of the value that I see the first black pixel at for one row 
    end = [] # a list of the values that I see the last black pixel at for one row 
    all_starts = [] # a list off all the starts for all the rows 
    all_ends = [] # a list of all teh ends for all the rows 
    threshold_val = 40 # the value that there must be at least that many consecutive pixels for it to be seen as tree and not a mistake 

    # measure the amount of black in each row 
    for i,x in enumerate(closing):
        # going across the row 
        for j,y in enumerate (x): 
            #going down the image
            if y == 255:
                #means it is white 
                if consecutive >= threshold_val: 
                    #the pixels have been added to the list 
                    end.append(j-1)
                
                consecutive = 0 
            else: 
                #means it is black
                consecutive += 1
                if consecutive == threshold_val: 
                    #if there has been 30 black pixels in a row at that to the black_pixel count 
                    total_black += threshold_val
                    start.append(j-39)
                elif consecutive > threshold_val: 
                    total_black += 1 
                if j == (len(x)-1):
                    end.append(j)
        black_pixels.append(total_black)
        if start != []:
            all_starts.append(start)
        if end != []:
            all_ends.append(end)
        total_black = 0 # reset total black pixel count for the next row 
        consecutive = 0 # reset consecutive count for the next row 
        start = [] # reset the list of start values for the next row  
        end = [] # reset the list of end values for the next row 

    all_middle = [] #initializes list of all the middle points in each row 
    for i in range (len(all_starts)):
        #calculate the middle based on the start and values in each row 
        mid = round((all_ends[i][0] - all_starts[i][0])/2 +all_starts[i][0])
        all_middle.append(mid) 

    
    final_img = copy.deepcopy(closing) 

    middle_line = statistics.mode(all_middle)

    # calculate the diameter of each row  
    diameter_estimates = [] # imitialize diameter estimate list 
    for idx in range (len (all_starts)):
        diameter = all_ends[idx][0] - all_starts[idx][0]
        diameter_estimates.append(diameter)

    #reorganize list so the list is organized by least seen number to most seen numbers 
    diameter_estimates_occurances = check_occurances(diameter_estimates)
    diameter_estimates_sorted = create_list(diameter_estimates_occurances)
    #change list so only the top 40% most seen values are in the list 
    per = 0.6
    length_from_end = int(len(diameter_estimates_sorted) * (per))
    diameter_estimates_new = diameter_estimates_sorted[-length_from_end:-1]
    #Average the top 40% vaues 
    avg_new = np.average(diameter_estimates_new)
    #average of all the values to compare 
    avg = np.average(diameter_estimates)

    print(f"avg: {avg}  avg_new: {avg_new}")
    

    #reorganize list so the list is organized by least seen number to most seen numbers
    middle_estimates_occurances = check_occurances(all_middle)
    middle_estimates_sorted = create_list(middle_estimates_occurances)
    #change list so only the top 40% most seen values are in the list 
    per = 0.6
    length_from_end = int(len(middle_estimates_sorted) * (per))
    middle_estimates_new = middle_estimates_sorted[-length_from_end:-1]
    #Average the top 40% vaues
    avg_new_middle = np.average(middle_estimates_new)
    #average of all the values to compare 
    avg_middle = np.average(all_middle)

    print(f"avg: {avg_middle}  avg_new: {avg_new_middle}, mode: {middle_line}")

    #create a box using the middle line and the radius 
    create_box (avg_new/2, avg_new_middle, final_img)
        
    rows = final_img.shape[0]
    columns = final_img.shape[1]
    mult = np.zeros((rows*2, columns*2))
    print(columns)
    if avg_new_middle < (columns/2 -5):
        in_front = False 
        direction = "left"
    elif avg_new_middle > (columns/2 +5):
        in_front = False 
        direction = "right"
    else:
        in_front = True 
        direction = None
        

    pic1 = frames[frames_using]
    pic1 = F.to_pil_image(pic1.to("cpu"))
    pic2 = flow_imgs[0]
    pic2 = F.to_pil_image(pic2.to("cpu"))
    pic3 = closing
    pic4 = final_img
    pic5 = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    #https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    if in_front: 
        color = (0,255,0)
    else: 
        color = (255, 0,0)
    image = cv2.putText(pic5, '132', (int(avg_new_middle - avg_new - 50) ,240), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, color, 3,  cv2.LINE_AA)
    image = cv2.putText(pic5, '132', (int(avg_new_middle + avg_new + 50 ),240), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, color, 3,  cv2.LINE_AA)
    image = cv2.putText(pic5, direction, (300,400), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, color, 3,  cv2.LINE_AA)
    
    fig=plt.figure(frame)
    fig.add_subplot(2,2,1)
    plt.imshow(pic1)
    fig.add_subplot(2,2,2)
    plt.imshow(pic2)
    fig.add_subplot(2,2,3)
    plt.imshow(pic3, cmap='gray')
    #fig.add_subplot(2,2,4)
    #plt.imshow(pic4, cmap = 'gray')
    fig.add_subplot(2,2,4)
    plt.imshow(pic5)
    plt.show()
    
    pic1 = frames[frame][0].numpy()
    pic2 = flow_saved
    image = cv2.putText(pic4, '132', (int(avg_new_middle - avg_new - 50) ,240), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,155,0), 3,  cv2.LINE_AA)
    image = cv2.putText(pic4, '132', (int(avg_new_middle + avg_new + 50 ),240), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0,155,0), 3,  cv2.LINE_AA)
    image = cv2.putText(pic4, direction, (300,400), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (255,0,0), 3,  cv2.LINE_AA)
    for x in range (rows*2):
        for y in range (columns*2): 
            if x<rows: 
                if y < columns: 
                    mult[x][y] = pic1[x][y]
                else: 
                    mult[x][y] = pic2[x][y-columns]
            else: 
                if y < columns: 
                    mult[x][y] = pic3[x-rows][y]
                else: 
                    mult[x][y] = pic4[x-rows][y-columns]
                    
    plt.imshow(mult, cmap='gray')
    plt.show()



