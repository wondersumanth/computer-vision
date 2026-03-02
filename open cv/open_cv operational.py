### Basic operations with OpenCV
# using Python

import cv2

# Operation-1: Read the image from directory
image = cv2.imread("image.jpg")
# ..................................

# Operation-2: Blur the image
blurred = cv2.blur(image, (50, 50))
#..................................

# Operation-3: Visualize the image
# cv2.imshow("test", blurred)
# cv2.waitKey(0)  # display until key press
#........................

# Operation-4: Write the image in directory
# cv2.imwrite("test.png", blurred)
# ...................

# Operation-5: Crop the image and display it
# sliced_image = image[240:450, 680:900]
# cv2.imshow("sliced_window", sliced_image)
# cv2.waitKey(0)
# .................................


# Operation-6: Read and process video file
# videocapture = cv2.VideoCapture("video.mov")

# while videocapture.isOpened():
#     success, frame = videocapture.read()
#     if success:
#         cv2.imshow("test", frame)
#         if cv2.waitKey(2) & 0xFF == ord("q"):
#             break
#     else:
#         break  # break the window once frames completed
# .......................


import os

# Operation-6: Read video and save frames every 0.5 seconds
videocapture = cv2.VideoCapture("video.mov")

if not videocapture.isOpened():
    print("Error opening video file")
    exit()

# Get FPS of video
fps = videocapture.get(cv2.CAP_PROP_FPS)
print("Video FPS:", fps)

# Calculate frame interval for 0.5 seconds
frame_interval = int(fps * 0.5)

# Create output directory
output_folder = "extracted_frames"
os.makedirs(output_folder, exist_ok=True)

frame_count = 0
saved_count = 0

while videocapture.isOpened():
    success, frame = videocapture.read()
    if not success:
        break

    # Save frame every 0.5 seconds
    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
        # Ensure the filename is correctly handled for the current working directory
        cv2.imwrite(filename, frame)
        saved_count += 1

    cv2.imshow("Video Playback", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

videocapture.release()
cv2.destroyAllWindows()

print(f"Total frames saved: {saved_count}")
