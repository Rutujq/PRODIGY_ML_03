import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -----------------------------
# Load and preprocess images
# -----------------------------
image_data = []
image_labels = []

categories = ["cats", "dogs"]

for category in categories:
    folder_path = os.path.join("dataset", category)
    label = categories.index(category)

    for file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue  # skip invalid images

            img = cv2.resize(img, (64, 64))  # resize for uniformity
            image_data.append(img.flatten())  # convert to 1D array
            image_labels.append(label)

        except Exception as e:
            print("Error loading image:", e)

# Convert to numpy arrays
X = np.array(image_data)
y = np.array(image_labels)

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train SVM model
# -----------------------------
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")

# -----------------------------
# Test with one sample image
# -----------------------------
try:
    # Automatically pick first cat image
    test_image_path = os.path.join("dataset", "cats", os.listdir("dataset/cats")[0])

    test_img = cv2.imread(test_image_path)

    if test_img is None:
        print("❌ Could not load test image.")
    else:
        resized_img = cv2.resize(test_img, (64, 64))
        flat_img = resized_img.flatten().reshape(1, -1)

        prediction = svm_model.predict(flat_img)

        if prediction[0] == 0:
            result = "Cat"
        else:
            result = "Dog"

        print("Prediction:", result)

        # -----------------------------
        # Show image with prediction
        # -----------------------------
        plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Prediction: {result}")
        plt.axis('off')
        plt.show()

except Exception as e:
    print("Error during prediction:", e)