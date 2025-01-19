import json 

file = '/home/ryan/ros2_ws_groundtruth/src/Ground_Truth_Ros/ground_truth/json_files/UFO_Test_1_0_0_2025_01_19_10_30_25.json'

with open(file, 'r') as file:
    data = json.load(file)

#print(data)
W1_05 = []
W2_05 = []
Mean_05 = []
Median_05 = []

for key, value in data.items():
    if (float(value['measured diameter']) == 0.5):
        W1_05.append(value['W1 Diameter'])
        W2_05.append(value['W2 Diameter'])
        Mean_05.append(value['Mean Diameter'])
        Median_05.append(value['Median Diameter'])

print(f"W1 Diameter 0.5: {W1_05}")
print(f"W2 Diameter 0.5: {W2_05}")
print(f"Mean Diameter 0.5: {Mean_05}")
print(f"Median Diameter 0.5: {Median_05}")


