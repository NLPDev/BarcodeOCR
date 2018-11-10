from __future__ import print_function
import pyzbar.pyzbar as pyzbar

import numpy as np
import cv2
import csv
import sys

import pytesseract.pytesseract as pt

def decode(im):
    # Find barcodes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects


def lcs(X, Y, m, n):
    if m == 0 or n == 0:
        return 0;
    elif X[m - 1] == Y[n - 1]:
        return 1 + lcs(X, Y, m - 1, n - 1);
    else:
        return max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n));
    #
# # Display barcode
# def display(im, decodedObjects):
#     # Loop over all decoded objects
#     for decodedObject in decodedObjects:
#         points = decodedObject.polygon
#
#         # If the points do not form a quad, find convex hull
#         if len(points) > 4:
#             hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
#             hull = list(map(tuple, np.squeeze(hull)))
#         else:
#             hull = points;
#
#         # Number of points in the convex hull
#         n = len(hull)
#
#         # Draw the convext hull
#         for j in range(0, n):
#             cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)
#
#     # Display results
#     cv2.imshow("Results", im);
#     cv2.waitKey(0);
#

# Main 
if __name__ == '__main__':
    # Read image

    if len(sys.argv)!=3:
        print("Input Image file and Output csv")
    else:

        im = cv2.imread(sys.argv[1])

        decodedObjects = decode(im)

        row=['1', 'fasd']
        list=row
        list.append(row)
        list.append(row)
        # display(im, decodedObjects)
        with open(sys.argv[2], 'w') as writeFile:
             writer=csv.writer(writeFile)
             writer.writerow(['1', 'Account'])
             writer.writerow(['1', 'Account'])

        writeFile.close()

        img = cv2.imread(sys.argv[1])
        # Get text in the image
        text = pt.image_to_string(img)

        tt=text.split(' ')

        for aa in tt:
            print(aa)

