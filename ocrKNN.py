import numpy as np
# import cv2 as cv
from matplotlib import pyplot as plt
# img = cv.imread('digits.png')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # Now we split the image to 5000 cells, each 20x20 size
# cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# # Make it into a Numpy array. It size will be (50,100,20,20)
# x = np.array(cells)
# # Now we prepare train_data and test_data.
# train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
# # Create labels for train and test data
# k = np.arange(10)
# train_labels = np.repeat(k,250)[:,np.newaxis]
# test_labels = train_labels.copy()
# # Initiate kNN, train the data, then test it with test data for k=1
# knn = cv.ml.KNearest_create()
# knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
# ret,result,neighbours,dist = knn.findNearest(test,k=5)
# # Now we check the accuracy of classification
# # For that, compare the result with test_labels and check which are wrong
# matches = result==test_labels
# correct = np.count_nonzero(matches)
# accuracy = correct*100.0/result.size
# print( accuracy )
# # save the data
# np.savez('knn_data.npz',train=train, train_labels=train_labels)
#

import cv2
from imutils.video import VideoStream
import time
import imutils

with np.load('knn_data.npz') as data:
    print
    data.files
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

cap=cv2.VideoCapture(0)
time.sleep(2.0)

while(True):
    # Capture frame-by-frame
    ret1, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame

    frame = cv2.medianBlur(frame, 5)

    ret2, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

    # src.create(rows, cols, CV_8UC1);
    # src = imread(your - file, CV_8UC1);


    # source_thresh = cv2.adaptiveThreshold(frame, 255, 0, 1, 5, 2)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    filter = cv2.createBackgroundSubtractorMOG2()
    frame = filter.apply(frame)

    # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    frame = cv2.bitwise_not(frame, cv2.COLOR_GRAY2RGB)
    im2, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, 2, (0, 255, 0), 3)

    cv2.imshow('thresh', frame)

    test_img = cv2.resize(frame, (20, 20))
    x = np.array(test_img)
    test_img = x.reshape(-1, 400).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(test_img, k=1)
    # Print the predicted number
    print(int(result[0]))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#
# test_img = cv2.imread("test.jpg")
# test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# test_img = cv2.resize(test_img, (20, 20))
# x = np.array(test_img)
# test_img = x.reshape(-1, 400).astype(np.float32)
# ret, result, neighbours, dist = knn.findNearest(test_img, k=1)
# # Print the predicted number
# print(int(result))
