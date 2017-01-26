__author__ = 'jonathan'

import argparse
import cv2
import imutils
from imutils import auto_canny



foldername ="prueba/"



def good_circle(b):
    (x,y,w,h) = b
    ratio = float(w)/float(h)
    if (0.8 < ratio and ratio < 1.2):
        return True
    else:
        return False


def detect_pocillos(image,minradius):
    image = imutils.resize(image, width=600)
    shifted = cv2.pyrMeanShiftFiltering(image, 11, 21)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    (h,s,v) = cv2.split(cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV))

    cv2.imshow("Output", v)
    cv2.waitKey(0)
    (T, thresh)  = cv2.threshold(v,160,255,cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    for c in cnts:
        b = cv2.boundingRect(c)
        if(good_circle(b)):
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if(radius>minradius and radius < 30):
                print(radius)
                names = args["image"].split("/")
                names = names[1].split(".")[0]

                imgdisk = image[int(y) - int(radius):int(y) + int(radius), int(x) - int(radius):int(x) + int(radius)] #Recordar van primero las coordenadas y
                cv2.imwrite(foldername+names+"ImgDisk"+str(i)+".tif",imgdisk)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                i+=1



    cv2.imshow("Output", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image",required=False,help="Path to the image",default="Fotos/DSC_0007.jpg")#"images/a1.tif"
    ap.add_argument("-m","--minradius",required=False,default=15,type=int)
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    minradius = args["minradius"]
    detect_pocillos(image,minradius)
