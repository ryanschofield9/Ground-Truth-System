import numpy as np
def calculate_col_scores(submatrix):
    col_sum = submatrix.sum(axis = 0)  
    col_sum = col_sum /255.0
    return col_sum

submatrix = np.array([[0,255,0,255,0],[255,0,255,0,255],[0,255,0,255,0],[255,0,255,0,255],[0,255,0,255,0],[255,0,255,0,255]])
rows = submatrix.shape[0]
columns = submatrix.shape[1]
col_sum = calculate_col_scores(submatrix)
cumulative_sum = np.cumsum(col_sum)
middle = 2
diameter = 2
left_most = int(middle - diameter/2)
right_most = int(middle + diameter/2)
if left_most == 0: 
    white_val_inbox = cumulative_sum[right_most]-0
else: 
    white_val_inbox = cumulative_sum[right_most]-cumulative_sum[left_most-1]
black_val_outbox = ((columns-(right_most -left_most +1)) * rows) - (cumulative_sum[-1] - white_val_inbox)

print(f"White pixels in the box: {white_val_inbox}")
print(f"Black pixels outside the box: {black_val_outbox}")