import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

predicted_values = []
actual_values = []

current_dir = os.getcwd()
folder_path = os.path.join(current_dir, 'Raw_Data_1')  # specify the path to the folder
for filename in os.listdir(folder_path):  # loop through all files in the folder
    if filename.endswith(".txt"):  # check if the file is a text file
        file_path = os.path.join(folder_path, filename)  # get the full path to the file
        print(filename)
        with open(file_path) as f:  # open the file
            lines = f.readlines()[1:]  # read all lines from second line onwards
            # create an empty list to store the values
            for line in lines:
                split_line = line.split()  # split the line by whitespace
                if len(split_line) >= 3:  # check if there are at least 3 values in the line
                    predicted_values.append(int(split_line[2]))

                if "With_Baby" in filename:
                    actual_values.append(2)
                elif "Without_Baby" in filename:
                    actual_values.append(1)

# Compute confusion matrix
cm = confusion_matrix(actual_values, predicted_values)

# Write confusion matrix result to a CSV file
output_file = 'Results/confusion_matrix.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['True Negative', 'False Positive', 'False Negative', 'True Positive'])
    for row in cm:
        writer.writerow(row)

# Create ConfusionMatrixDisplay object and plot the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Compute F1 score value
f1_value = f1_score(actual_values, predicted_values)

# Calculate TP, FP, FN, TN, TPR, TNR, FPR, FNR
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FPR = FP / (FP + TN)
FNR = FN / (FN + TP)
ACC = (TP + TN) / (TP + FP + FN + TN)
print(f"True Positive (TP): {TP:.2f}")
print(f"True Negative (TN): {TN:.2f}")
print(f"False Positive (FP): {FP:.2f}")
print(f"False Negative (FN): {FN:.2f}")
print(f"True Positive Rate (TPR): {TPR:.2f}")
print(f"True Negative Rate (TNR): {TNR:.2f}")
print(f"False Positive Rate (FPR): {FPR:.2f}")
print(f"False Negative Rate (FNR): {FNR:.2f}")
print(f"Accuracy (ACC): {ACC:.2f}")
print(f"F1 Score: {f1_value:.2f}")
