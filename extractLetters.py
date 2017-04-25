import numpy as np
import cv2
import glob
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--disks" ,required=True, help = "path to the directory that contains our disks",default="discos/")
args = vars(ap.parse_args())

def print_contour(c,wI,hI):
    (x, y, w, h) = c
    print("x={},y={},w={},h={},wI={},hI={},r={}".format(x,y,w,h,wI,hI,(float)(w)/(float)(h)))

def good_contour(c,wI,hI):
    (x, y, w, h) = c
    if((h+y > 0.95*hI) or (w+x > 0.95*wI) or (y < 0.05*hI) or (x < 0.05*wI)
       or ((w<.1*wI) and (h < .1*hI))):
        return False
    return True

def lowers_than_1(cnts):
    n=0
    for c in cnts:
        (x, y, w, h) =  cv2.boundingRect(c)
        r = (float)(w)/(float)(h)
        if(r<1):
            n += 1
    if(n>1):
        return False
    return True

def pickBand(cnts,lower=True):
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        r = (float)(w) / (float)(h)
        if ((lower and r < 1) or (not(lower) and r>1) and (not(r==  1))):
            return c

for path in glob.glob(args["disks"] + "/*.tif"):
    name = path[path.rfind("/") + 1:]
    image = cv2.imread(path)
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    (h,s,v) = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    vCopy = v.copy()
    (w,h) = image.shape[:2]
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.circle(mask, (w/2, h/2), (int)(np.minimum(w,h)/2*.95), 255, -1)
    v = cv2.bitwise_and(v, v, mask=mask)
    v=cv2.adaptiveThreshold(v,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,5)
    cv2.imshow("im", v)
    cv2.waitKey(0)
    (cnts,_)=cv2.findContours(v.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    charCandidates = np.zeros(v.shape, dtype="uint8")
    #imageC = image.copy()

    goodContours=[]
    for c in cnts:
        b = cv2.boundingRect(c)
        print_contour(b,w,h)

        if good_contour(b,w,h):
            goodContours.append(c)

    for c in goodContours:
        hull = cv2.convexHull(c)
        cv2.drawContours(charCandidates,[hull],-1,255,-1)

    imageMask = cv2.bitwise_or(image, image, mask=charCandidates)
    rutaDest = "discosLetras/"+path.split("/")[1]

    if not os.path.isdir(rutaDest):
        os.mkdir(rutaDest,0o777)

    cv2.imwrite(rutaDest +"/"+ name, imageMask)
    cv2.imshow("image", imageMask)
    cv2.waitKey(0)
