#https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html#sphx-glr-auto-examples-others-plot-optical-flow-py

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
plt.rcParams["savefig.bbox"] = "tight"
weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()

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
    plt.show()
    plt.tight_layout()

def preprocess(img1_batch, img2_batch):
    #img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    #img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)

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
    radius_estimates_sorted = []
    for key in dict: 
        times = dict[key]
        for x in range (times):
            radius_estimates_sorted.append(key)
    return radius_estimates_sorted

def create_box (radius, middle, img):
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


video_path = "c:\\Users\\ryana\\Documents\\Robotics Research\\video_trial.avi"
frames, _, _ = read_video(str(video_path), output_format="TCHW")
'''
img1_batch = torch.stack([frames[60], frames[100]])
img2_batch = torch.stack([frames[61], frames[101]])
plot(img1_batch)
img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")
predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)

'''
#trying to do it for the whole video 
batch1 = []
batch2 = []
frames_using = 75
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
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")
predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")
flow_imgs = flow_to_image(predicted_flows)

# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)  

#try to threshold 
'''
img = flow_imgs #flow_imgs[0][0][0] 
#print(img)
flow_imgs2 = copy.deepcopy(flow_imgs)
#threshold_val = 0.90*max(flow_imgs[0][0][0])
for i,x in enumerate(img):
    for j,y in enumerate(x):
        for h, z in enumerate(y):
            threshold_val = 0.90*max(z)
            for g,pix in enumerate(z):
                if pix < threshold_val:
                    flow_imgs2[i][j][h][g] = 0
                else: 
                    flow_imgs2[i][j][h][g] = 255
print(flow_imgs2)
grid = [[img1, flow_img, flow_img2] for (img1, flow_img, flow_img2) in zip(img1_batch, flow_imgs, flow_imgs2)]
plot(grid)  
'''
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
#opencv threshold
img2 = flow_imgs[0].numpy()
flow_saved = copy.deepcopy(img2)
flow_saved = flow_saved[0]
img2 = np.moveaxis(img2, 0, -1)
#normalize image
img2 = (img2 - min(img2.reshape(-1)))/(max(img2.reshape(-1))-min(img2.reshape(-1)))*255
img2 = img2.astype(np.uint8)
print(img2.shape, min(img2.reshape(-1)), max(img2.reshape(-1)))
bwimg = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
plt.imshow(bwimg, cmap='gray')
plt.show()
#blurring image 
blur = cv2.GaussianBlur(bwimg,(5,5),0)
ret, img=cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(ret)
plt.imshow(img, cmap='gray')
plt.show()

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)


plt.imshow(closing, cmap='gray')
plt.show()
consecutive = 0 
total_black = 0
black_pixels = [] 
start = []
end = []
all_starts = []
all_ends = []

# measure the amount of black in each row 
for i,x in enumerate(closing):
    # going across the row 
    for j,y in enumerate (x): 
        #going down the image
        if y == 255:
            #means it is white 
            if consecutive >= 40: 
                #the pixels have been added to the list 
                end.append(j-1)
            
            consecutive = 0 
        else: 
            #means it is black
            consecutive += 1
            if consecutive == 40: 
                #if there has been 30 black pixels in a row at that to the black_pixel count 
                total_black += 40
                start.append(j-39)
            elif consecutive > 40: 
                total_black += 1 
            if j == (len(x)-1):
                end.append(j)
    black_pixels.append(total_black)
    if start != []:
        all_starts.append(start)
    if end != []:
        all_ends.append(end)
    total_black = 0 
    consecutive = 0 
    start = []
    end = [] 


all_middle = [] 
for i in range (len(all_starts)):
    mid = round((all_ends[i][0] - all_starts[i][0])/2 +all_starts[i][0])
    all_middle.append(mid)

plt.hist(all_middle)
plt.show()

final_img = copy.deepcopy(closing) 
slope = (all_middle[0] - all_middle[1])
x_saved = all_middle[0]
for j in range ((len (all_ends)-1)):
    #print("xsaved: ", x_saved)
    x = x_saved
    y = j 
    x2 = all_middle[j+1]
    y2 = j+1
    #final_img = cv2.line(final_img ,(all_middle[j],j),(all_middle[j+1],j+1),(255,0,0),5)
    slope_new = x2-x 
    #print(f"slope new = {slope_new}, slope = {slope}")
    if abs(slope-slope_new) > 15  :
        #print("in back up function")
        x = x_saved
        #x2 = x_saved
        #print(f"x: {x} x2_old = {x2}")
        #x2 = round(x_saved + (x2-x_saved)/2)
        #print(f"x2_new: {x2}" )
        
        if slope_new > 0: 
            x2 = x_saved 
        else: 
            x2 = x_saved 
        x_saved = x2
        
    else: 
        slope = slope_new 
        x_saved = x2
    #final_img = cv2.line(final_img ,(x,y),(x2,y2),(255,0,0),5)
plt.imshow(final_img, cmap='gray')
plt.show()

#https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/
sns.histplot(all_middle, kde=True, color='lightgreen', edgecolor='red')
plt.show()

middle_line = statistics.mode(all_middle)
#final_img = cv2.line(final_img,(middle_line,0),(middle_line,640),(255,0,0),5)
plt.imshow(final_img, cmap='gray')
plt.show()

# calculate the radius of each line 
radius_estimates = []
for idx in range (len (all_starts)):
    radius = all_ends[idx][0] - all_starts[idx][0]
    radius_estimates.append(radius)
plt.hist(radius_estimates)
plt.show()
#https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/

sns.histplot(radius_estimates, kde=True, color='lightgreen', edgecolor='red')
plt.show()

radius_estimates_occurances = check_occurances(radius_estimates)
radius_estimates_sorted = create_list(radius_estimates_occurances)
#print(radius_estimates_sorted)
per = 0.3
length_from_end = int(len(radius_estimates_sorted) * (per*2))
radius_estimates_new = radius_estimates_sorted[-length_from_end:-1]
#print(radius_estimates_new)
avg_new = np.average(radius_estimates_new)
avg = np.average(radius_estimates)

print(f"avg: {avg}  avg_new: {avg_new}")
 


middle_estimates_occurances = check_occurances(all_middle)
middle_estimates_sorted = create_list(middle_estimates_occurances)
#print(middle_estimates_sorted)
per = 0.3
length_from_end = int(len(middle_estimates_sorted) * (per*2))
middle_estimates_new = middle_estimates_sorted[-length_from_end:-1]
#print(middle_estimates_new)
avg_new_middle = np.average(middle_estimates_new)
avg_middle = np.average(all_middle)
print(f"avg: {avg_middle}  avg_new: {avg_new_middle}, mode: {middle_line}")
create_box (avg_new/2, avg_new_middle, final_img)
    
'''
#just finding center of top and center of bottom and making a straightline 
avg = np.average(black_pixels)
line_start_x =int((all_ends[0][-1] - all_starts[0][0] )/2 + all_starts[0][0])
line_end_x = int(( all_ends[-1][-1] - all_starts[-1][0])/2 + all_starts[-1][0])

#find slope from bottom left to top left, bottom left to middle, middle to top left and compare them to the three found on the right side

#bottom to top left slope 
btl = (all_starts[-1][0] - all_starts[0][0])/len(all_starts)

#bottom to middle left slope 

bml = (all_starts[-1][0] - all_starts[round(len(all_starts)/2)][0]) / round(len(all_starts)/2)

#middle to top left slope 
mtl = (all_starts[round(len(all_starts)/2)][0]-all_starts[0][0])/ round(len(all_starts)/2)

print ("btl: ", btl, " bml: ", bml, " mtl: ", mtl)

#bottom to top right slope 
btr = (all_ends[-1][-1] - all_ends[0][-1])/len(all_ends)

#bottom to middle right 
bmr = (all_ends[-1][-1] - all_ends[round(len(all_ends)/2)][-1]) / round(len(all_ends)/2)

#middle to top left slope 
mtr = (all_ends[round(len(all_ends)/2)][-1]-all_ends[0][-1])/ round(len(all_ends)/2)

print ("btr: ", btr, " bmr: ", bmr, " mtr: ", mtr)

#use slopes to figure out what the correct average slope is 

#left slope differences 
left = [btl, bml, mtl]
avg_l = np.average(left)
dif_l = [abs(btl - avg_l), abs(bml - avg_l), abs(mtl - avg_l)]

#right slope differences
right = [btr,bmr, mtr]
avg_r = np.average(right)
dif_r = [abs(btr - avg_r), abs(bmr - avg_r), abs(mtl - avg_r)]

print("dif_l: ", dif_l, " dif_r: ", dif_r)
avgs = []
bottom_tally = 0 
top_tally = 0 
max = 0 
spot = 0 
for i in range (1, len(left)): 
    if dif_l[i]< 0.1:
        avgs.append(left[i])
        if dif_l[i] > max: 
            max = dif_l[i]
            spot = i 
    if abs(dif_r[i] < 0.1): 
        avgs.append(right[i])
        if dif_r[i] > max: 
            max = dif_r[i]
            spot = i 
print(avgs)
final_slope = np.average(avgs)
print("top_tally: ", top_tally, " bottom_tally: ", bottom_tally)
if spot == 1: 
    start_x = int(line_start_x ) 
    end_x = int(line_start_x - (len(all_starts)*final_slope))
else:
    end_x = int(line_end_x )  
    start_x = int(line_end_x - (len(all_starts)*final_slope))

print(f"start x: {start_x}  end_x: {end_x}")
#draw line on the image 
#https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
#img = cv2.line(closing ,(line_start_x,0),(line_end_x,640),(255,0,0),5)
img = cv2.line(img ,(start_x,640),(end_x,0),(255,0,0),5)
'''

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

'''
pic1 = frames[120][0].numpy()
pic2 = flow_saved
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
                 
plt.imshow(mult)
plt.show()
'''
fig=plt.figure()
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


# open/close image after thresholding 
#blurr image before thresholding 
#using pca fit a line 


