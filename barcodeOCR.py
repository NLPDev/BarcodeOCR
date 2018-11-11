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
    # for obj in decodedObjects:
    #     print('Type : ', obj.type)
    #     print('Data : ', obj.data, '\n')

    return decodedObjects


def lcs(X, Y, m, n):
    l = max(m, n) + 5

    res = [[0 for x in range(l)] for y in range(l)]
    res[0][0] = 0

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                res[i + 1][j + 1] = 1 + res[i][j]
            else:
                res[i + 1][j + 1] = max(res[i + 1][j], res[i][j + 1])

    return res[m][n]
# Display barcode
def display(im, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points;

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im);
    cv2.waitKey(0);


# Main 
if __name__ == '__main__':
    # Read image

    if len(sys.argv)!=3:
        print("Input Image file and Output csv")

    else:

        im = cv2.imread(sys.argv[1])

        decodedObjects = decode(im)
        # display(im, decodedObjects)

        row=['1', 'fasd']
        list=row
        list.append(row)
        list.append(row)
        # display(im, decodedObjects)


        img = cv2.imread(sys.argv[1])
        # Get text in the image
        text = pt.image_to_string(img)

        tt=text.split(' ')
        flag=[0]


        for aa in tt:
            flag.append(0)

        i=0

        with open(sys.argv[2], 'w') as writeFile:
            writer=csv.writer(writeFile)

            writer.writerow(['SN_Barcode_Value', 'SN_OCR_Text_Value'])

            for bb in decodedObjects:
                aa = bb.data
                cc = str(aa)
                cc = cc.split('\'')
                cc = cc[1]
                j = -1
                maxlen = -1
                mnum = -1
                for pp in tt:
                    cmp=""
                    for pi in range(len(pp)):
                        if pp[pi].isdigit() or pp[pi].isupper():
                            cmp=cmp+pp[pi]

                    cmp=cmp+'\0'
                    j = j + 1

                    if flag[j] == 1 or len(cmp)>30:
                        continue


                    ml = lcs(cc, cmp, len(cc), len(cmp))
                    if maxlen < ml:
                        maxlen = ml
                        mnum = j

                flag[mnum] = 1
                writer.writerow([cc, tt[mnum]])





        writeFile.close()




