import imutils
import numpy as np
import cv2
import argparse
from os import listdir

rutaCarpetaDefecto="discosSegmented/"
tipoDisco="TOB-10/"


#Por si hay que tratar alguna imagen en especial
'''
image = cv2.imread(rutaCarpetaDefecto+tipoDisco+"20160527_125843ImgDisk5.tif")
rotated = imutils.rotate(image, 270)
cv2.imshow("Rotated by 90 Degrees", rotated)

cv2.imwrite(rutaCarpetaDefecto + tipoDisco + "mod0"  + ".tif", rotated)
cv2.waitKey(0)
'''


#Esto es para oscurecer las imagenes y modificarlas un poco, en ocasiones la letra se ve mejor.
#Mediante la ecualizacion del hostograma despues de haber hecho subtract comprobamos como las letras se pueden diferenciar mejor

i=1;
for cosa in listdir(rutaCarpetaDefecto+tipoDisco):
    print cosa;
    image = cv2.imread(rutaCarpetaDefecto+tipoDisco+cosa);
    M = np.ones(image.shape, dtype="uint8") * 50;
    subtracted = cv2.subtract(image, M);
    image = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(image)
    #cv2.imshow("Subtracted", subtracted);
    cv2.imwrite(rutaCarpetaDefecto + tipoDisco + "subtract"+str(i)  + ".tif", eq);
    #cv2.waitKey(0);
    i+=1;


#Esto es para rotar las fotos

i=1;
for cosa in listdir(rutaCarpetaDefecto+tipoDisco):
    print cosa;
    image = cv2.imread(rutaCarpetaDefecto+tipoDisco+cosa);
    rotated = imutils.rotate(image, 30);
    #cv2.imshow("Rotated by 90 Degrees", rotated);
    cv2.imwrite(rutaCarpetaDefecto + tipoDisco + "rot"+str(i)  + ".tif", rotated);
    #cv2.waitKey(0);
    i+=1;


#Esto es para hacer translaciones de las imagenes

i=1;
for cosa in listdir(rutaCarpetaDefecto+tipoDisco):
    print cosa;
    image = cv2.imread(rutaCarpetaDefecto+tipoDisco+cosa);
    shifted = imutils.translate(image, 15, 0);
    #cv2.imshow("Shifted Down", shifted);
    cv2.imwrite(rutaCarpetaDefecto + tipoDisco + "tras"+str(i)  + ".tif", shifted);
    #cv2.waitKey(0);
    i+=1;

