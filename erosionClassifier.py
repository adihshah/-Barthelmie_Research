from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt
from functools import reduce


img = cv2.imread('blade1.jpg')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
ret, label, center = cv2.kmeans(Z, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

label = label.flatten()

count = 0
label_to_discount = label[0]
for l in label:
	if l != label_to_discount:
		count = count+1

gray = cv2.imread('blade1.jpg', 0)

m, n = gray.shape





for i in range(m):
    for j in range(n):
        if gray[i][j] < 50:
            gray[i][j] = 255
        else:
           gray[i][j] = 255 - gray[i][j]



blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")

# loop over the unique components
sum = 0
for label in np.unique(labels):
    if label == 0:
        continue

    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 300:
        print(numPixels)
        
        mask = cv2.add(mask, labelMask)
        sum += (numPixels/count)*100

if sum > 15:
    print("The image has major delaminations")
elif sum > 5:
    print("The image has minor delaminations")
elif sum >0:
    print("The image has gouges")
else:
    print("The image has pits")

plt.plot(122), plt.imshow(mask, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()




