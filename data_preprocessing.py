from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import cv2
import pdb
import numpy as np
from skimage.feature import hog
from skimage import exposure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

num_of_authors = 2
max_num_of_words_per_author = 4
image_size = (64, 64)  # Desired image size for preprocessing

# Lists to store features and labels
features_list = []
labels_list = []

# Collect HOG features and labels
for author_no in range(num_of_authors):
    file_desc_name = "author" + str(author_no + 1) + "/word_places.txt"
    file_desc_ptr = open(file_desc_name, 'r')
    text = file_desc_ptr.read()
    lines = text.split('\n')
    number_of_lines = len(lines) - 1

    for i in range(number_of_lines):
        row_values = lines[i].split()
        
        if row_values[0] != '%':
            image_file_name = "author" + str(author_no + 1) + "\\" + row_values[0][1:-1]
            image = mpimg.imread(image_file_name)
            word = row_values[1]
            row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                int(row_values[4]), int(row_values[5])
            
            # Ensure coordinates are within bounds
            row1 = max(0, min(row1, image.shape[0]))  # Ensure row1 is within [0, image height]
            row2 = max(0, min(row2, image.shape[0]))  # Ensure row2 is within [0, image height]
            column1 = max(0, min(column1, image.shape[1]))  # Ensure column1 is within [0, image width]
            column2 = max(0, min(column2, image.shape[1]))  # Ensure column2 is within [0, image width]
            
            # Check if subimage is empty
            if row1 < row2 and column1 < column2:
                subimage = image[row1:row2, column1:column2] 
                
                # Resize the image to a fixed size
                subimage_resized = cv2.resize(subimage, image_size)
                
                # Convert the image to grayscale
                subimage_gray = cv2.cvtColor(subimage_resized, cv2.COLOR_RGB2GRAY)
                
                # Extract HOG features
                features = hog(subimage_gray, orientations=9, pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2), block_norm='L2-Hys')
                
                # Append features and labels to the lists
                features_list.append(features)
                labels_list.append(author_no)  # Label each image with its author number

    file_desc_ptr.close()

# Convert lists to numpy arrays
X = np.array(features_list)
y = np.array(labels_list)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
