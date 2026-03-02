"""
Advanced OpenCV Hands-On Applications
For Participants – Practical Computer Vision Exercises
"""

import cv2
import numpy as np

# -------------------------------------------
# 1️⃣ Read and Display Image
# -------------------------------------------
image = cv2.imread("image_boys.png")

if image is None:
    print("Error: Image not found.")
    exit()

cv2.imshow("Original Image", image)
cv2.waitKey(0)


# -------------------------------------------
# 2️⃣ Convert to Grayscale
# Use Case: Preprocessing for Face Detection / OCR
# -------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)


# -------------------------------------------
# 3️⃣ Edge Detection (Canny)
# Use Case: Object Boundary Detection
# -------------------------------------------
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Edge Detection", edges)
cv2.waitKey(0)


# -------------------------------------------
# 4️⃣ Face Detection (Haar Cascade)
# Use Case: Attendance System / Security
# -------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Face Detection", image)
cv2.waitKey(0)


# -------------------------------------------
# 5️⃣ Color Detection (Detect Red Objects)
# Use Case: Traffic Signal Detection / Object Tracking
# -------------------------------------------
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Red Color Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
