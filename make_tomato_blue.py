import numpy as np
import cv2
import matplotlib.pyplot as plt

image = plt.imread("salad.jpg")

## K Means Clustering Color Data
vectorized = image.reshape((-1, 3))
vectorized = np.float32(vectorized)

termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 9
_, label, center = cv2.kmeans(vectorized, K, bestLabels=None, criteria=termination_criteria, attempts=10, flags=0)

k_image = np.uint8(center)[label.flatten()]
k_image = k_image.reshape((image.shape))

## Mask for reds
img_hsv = cv2.cvtColor(k_image, cv2.COLOR_RGB2HSV)

lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(img_hsv, lower_red, upper_red)
mask = np.dstack((mask, mask, mask))

output_image = np.copy(image)
output_image = np.where(mask==(0, 0, 0), output_image, 255 - output_image)
plt.imsave("blue_tomato.jpg", output_image)