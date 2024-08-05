import cv2
import numpy as np

img= cv2.imread(r'C:\Users\berra\Desktop\RKSoft\8\grape.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 1.5)
img = cv2.Canny(img, 50, 300)

cv2.imwrite(r'C:\Users\berra\Desktop\RKSoft\8\grape2.png', img)



params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 10
params.maxArea = 50
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)
img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
image_new= cv2.imwrite(r'C:\Users\berra\Desktop\RKSoft\8\grase4.jpg',img_with_keypoints)


h, w = img.shape[:]

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.5, int(w / 20), param1=50, param2=20, minRadius=int(w / 40),
maxRadius=int(w / 15))

circles = np.uint16(np.around(circles))

for c in circles[0, :]:
    print(c)

cv2.circle(circles, (c[0], c[1]), c[2], (0, 255, 0), 2)
cv2.circle(circles, (c[0], c[1]), 1, (0, 0, 255), 1)
cv2.imwrite(r'C:\Users\berra\Desktop\RKSoft\8\grape3.png',img_with_keypoints)


# import required libraries
import cv2
import numpy as np

# read input image
img= cv2.imread(r'C:\Users\berra\Desktop\RKSoft\8\grape.png')

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_purple = np.array([120, 50, 50])
upper_purple = np.array([140, 255, 255])

# Create a mask. Threshold the HSV image to get only yellow colors
mask = cv2.inRange(img, lower_purple, upper_purple)

# Bitwise-AND mask and original image
result = cv2.bitwise_and(img, img, mask= mask)

# display the mask and masked image
cv2.imwrite(r'C:\Users\berra\Desktop\RKSoft\8\grape6.png', mask)


import cv2 
import numpy as np 

# Let's load a simple image with 3 black squares 
image = cv2.imread(r'C:\Users\berra\Desktop\RKSoft\8\grape10.png') 

# Find Canny edges 
edged = cv2.Canny(image, 30, 200) 

# Finding Contours 
# Use a copy of the image e.g. edged.copy() 
# since findContours alters the image 
contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

# Draw all contours 
# -1 signifies drawing all contours 
cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
cv2.imwrite(r'C:\Users\berra\Desktop\RKSoft\8\grape12.png', image)
print("Number of Contours found = " + str(len(contours))) 
