# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import time

# Load the classifier
clf = joblib.load("digits_cls.pkl")

# Read the input image
im = cv2.imread("photo_2.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
aa, ctrs, hier= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]


    # # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))
    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()


#
# cap=cv2.VideoCapture(0)
# time.sleep(2.0)
#
# while(True):
#     # Capture frame-by-frame
#     ret1, frame = cap.read()
#
#     numf = frame
#
#     # Our operations on the frame come here
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#
#     # Display the resulting frame
#
#     frame = cv2.medianBlur(frame, 5)
#
#     ret2, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
#
#     frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
#     filter = cv2.createBackgroundSubtractorMOG2()
#     frame = filter.apply(frame)
#
#     frame = cv2.bitwise_not(frame, cv2.COLOR_GRAY2RGB)
#     im2, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     cv2.drawContours(frame, contours, 2, (0, 255, 0), 3)
#
#     digits={}
#     locs=[]
#     groupOutput=[]
#     output=[]
#     out=np.zeros(frame.shape, np.uint8)
#
#
#     for (i, c) in enumerate(contours):
#         (x, y, w, h)=cv2.boundingRect(c)
#         roi=frame[y:y+h, x:x+w]
#
#         locs.append((x, y, w, h))
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#
#
#         cv2.imshow('thresh', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         # Read the input image
#         # im = cv2.imread("photo_2.jpg")
#
#         im=numf
#
#         # Convert to grayscale and apply Gaussian filtering
#         im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#         im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
#
#         # Threshold the image
#         ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
#
#         # Find contours in the image
#         aa, ctrs, hier= cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
#         # Get rectangles contains each contour
#         rects = [cv2.boundingRect(ctr) for ctr in ctrs]
#
#         # For each rectangular region, calculate HOG features and predict
#         # the digit using Linear SVM.
#         for rect in rects:
#             # Draw the rectangles
#
#             cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
#             # Make the rectangular region around the digit
#             leng = int(rect[3] * 1.6)
#             pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
#             pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
#             roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
#
#
#             # # Resize the image
#             roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#             roi = cv2.dilate(roi, (3, 3))
#             # Calculate the HOG features
#             roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#             nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
#             cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
#
#         # cv2.imshow("Resulting Image with Rectangular ROIs", im)
#         cv2.imwrite("res.jpg", im)
#

#
#         break
#
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()