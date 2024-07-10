import numpy as np

def calculate_col_scores(closing):
    #add up the sum off the pixels in each row (essentially counting white pixels becuase either 255 or 0)
    col_sum = closing.sum(axis = 0) 
    #account for the fact that white pixels are 255 so the number of white pixels is seen easier 
    #col_sum = col_sum /255.0
    print(col_sum)
    print(len(col_sum))
    return col_sum

def score_options(diameters, middles, closing):
    rows = closing.shape[0]
    columns = closing.shape[1]
    col_sum = calculate_col_scores(closing)
    #print(col_sum)
    cumulative_sum = np.cumsum(col_sum)
    print(cumulative_sum)
    min_score = cumulative_sum[-1]
    #print(cumulative_sum)
    diameter_save = 0 
    middle_save = 0
    for diameter in diameters: 
        for middle in middles:  
            left_most = int(middle - diameter/2)
            right_most = int(middle + diameter/2)
            print(f"diameter: {diameter}  middle: {middle}  rightmost: {right_most}  leftmost: {left_most}")
            if left_most < 0 or right_most > 640:
                score =  cumulative_sum[-1]
            else: 
                #number of white pixels in the square 
                if left_most == 0: 
                    white_val_inbox = cumulative_sum[right_most]-0
                   
                else: 
                    white_val_inbox = cumulative_sum[right_most]-cumulative_sum[left_most-1]
                print(f"cumsum right:{cumulative_sum[right_most]}  cumsum left: {cumulative_sum[left_most-1]} ")
                #number of black pixels outside the squre 
                black_val_outbox = ((columns-(right_most -left_most +1)) * rows) - (cumulative_sum[-1] - white_val_inbox)
                score = white_val_inbox 
            if min_score > score: 
                min_score = score 
                diameter_save = diameter 
                middle_save = middle
            print(f"diameter: {diameter}  middle: {middle}  rightmost: {right_most}  leftmost: {left_most} score: {score}")
            print(f" black outside: {black_val_outbox}   white inside: {white_val_inbox}")
    return diameter_save, middle_save, min_score

closing = [ [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]
closing = np.array(closing)
score_options([2],[1], closing)