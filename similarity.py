from detectanddescribe import DetectAndDescribe
from diskMatcher import DiskMatcher
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score, precision_score
from sklearn.model_selection import train_test_split
import argparse
import cv2
import numpy as np
import pandas as pd
import os


 #Lo necesitamos poner aqui para que se guarden de forma global todas las imagenes,
# de la otra forma como se hace referencia de forma recursiva la variable se reinicia cada vez que se mete en una carpeta nueva
#En esta diccionario guardamos las rutas de las imagenes que han obtenido alto % de coincidencia con la que hemos pasado y el % que tienen.

def similarityImage(imageVector,img):  # funcion principal
    matchIm = {}
    maxItems=5 #Este es el numero maximo de elementos que vamos a permitir en el diccionario de los resultados

    dad = DetectAndDescribe(cv2.FeatureDetector_create("FAST"),
                            cv2.DescriptorExtractor_create("FREAK"))

    queryImage = cv2.imread(img) #Leemos la imagen pasada como parametro
    (_, _, v) = cv2.split(cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV))

    (queryKps, queryDescs) = dad.describe(v);  # En v tenemos la imagen que estamos comparando en cada iteracion con el resto de imagenes de la carpeta


    #for j in di:  # Recorremos el directorio, solo va a haber carpetas dentro por lo que j siempre sera una carpeta y asi evitamos comprobaciones de si es un archivo
    #   di2 = glob(j + "/" + "*") #Como dentro solo vamos a tener directorios con las clases de las imagenes, hacemos lo mismo que al principio
        # guardamamos la ruta de la carpeta en la que estamos en ese momento. En ese momento tenemo un array con todas las imagenes que estan dentro de esa carpeta
    cv = DiskMatcher(dad, imageVector, "BruteForce-Hamming") #Comparamos todas las imagenes dentro de la carpeta de la clase de ese momento
    # y le pasamos como parametro precisamente la carpeta de la clase actual (Ej: Ak-30) para hacer la comparacion
    results = cv.search(queryKps, queryDescs); #Guardamos el vector con todos los resultados de la comparacion de la imagen con las de dentro de la carpeta

    if len(results) != 0:
        for (i, (score, diskPath)) in enumerate(results): #Recorremos el vector de resultados para aniadir solo las que tengan un % mas alto,
             # el limite de elementos lo ponemos en maxItems
            if (score >= 0.65 ):
                if (len(matchIm) < maxItems): #Solo queremos las 5 fotos que mas se parezcan a nuestra imagen
                    matchIm[diskPath] = score;
                else:
                    minvalue = min(matchIm.values());
                    if (minvalue < score):
                        minkey = [key for key, value in matchIm.iteritems() if value == minvalue];  # Devuelve un vector con los valores minimos, puede ser que sean mas de uno
                        del matchIm[minkey[0]];  # borramos del diccionario el primero de los elementos que tenemos guardado en el vector
                        matchIm[diskPath] = score;

    '''
    Tenemos en matchIm las 5 imagenes que tienen mas parecido con la que le hemos pasado por parametro.
    Con ello ahora tenemos que decir de que tipo se
    predice que es mirando cual es la clase que mas se repite.

    Para eso crearemos un diccionario que tenga la clase como clave y el numero de apariciones como valor, asi devolveremos
    la clave que tenga mayor apariciones
    '''
    resultDic={}; #En esta variable guardaremos las rutas que han obtenido alto % de coincidencias y contaremos el numero de apariciones
    #para luego devolver la clase que tenga mayor numero de apariciones
    for pa, sc in matchIm.iteritems():
        route = pa.split("/");
        if (resultDic.has_key(route[1])):
            resultDic[route[1]]+=1;
        else: resultDic[route[1]]=1;

    if (len(resultDic) != 0):
        maxx = max(resultDic.values());  # cogemos el maximo del diccionario de resultados
        for k in resultDic.items():
            if maxx == k[1]:
                return k[0]
    else: return "Error"

'''
En este programa cogeremos lo que hace similarity y lo llevaremos a un ambito mas global. Ahora necesitamos saber el tipo
de todas las imagenes del data set. Tendremos que eliminar el caso de comparacion de la imagen que estamos usando en ese
momento consigo misma.
'''

def similarityDataSet(carp):
    di = glob(carp + "/" + "*")  # vemos todo lo que esta adentro del directorio que nos manden

    images=[]
    target_names = ['AK-30', 'AMC-30', 'AMP-10', "Error"]

    for j in di:
        images+=(glob(j + "/" + "*"))

    for i in range (0,10):
        vectorExpectedType = []
        vectorRealType = []
        (X_train,_,images_test,_)=train_test_split(images,np.zeros(len(images)),test_size=0.25,random_state=i)
        for j in X_train:
            vectorExpectedType.append(similarityImage(X_train,j))
            vectorRealType.append(j.split("/")[1])

        comprobeResults(vectorRealType, vectorExpectedType, target_names)



def comprobeResults(y_true, y_pred,target_names): #metodo para mostrar/guardar los datos despues de haber hecho la comparacion de las imagenes
    print "These are the predicted type of the images"
    print y_pred
    print "\n"
    print "These are the real type of the images"
    print y_true
    classi_rep=classification_report(y_true, y_pred, target_names=target_names)
    print(classi_rep)
    conf_mat=confusion_matrix(y_true, y_pred, labels=target_names)
    print conf_mat
    print "\n"
    print accuracy_score(y_true,y_pred)

    results = pd.DataFrame(
        {
            'classification_report':[classi_rep],
            'confusion_matrix':[conf_mat],
            'accuracy_score':[accuracy_score(y_true,y_pred)]
        }
    )
    if not os.path.isfile('resultados.csv'):
        results.to_csv('resultados.csv', index=False)
    else:  # else it exists so append without writing the header
        results.to_csv('resultados.csv', index=False, mode='a', header=False)



if __name__ == "__main__":  # Asi se ejecutan los scripts

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--disks", required=False, help="Path to the directory that contains our disks",
                    default="discPrueba")  # "images/a1.tif"
    ap.add_argument("-i", "--image", required=False, help="Path to the image that we want to compare",
                    default="20160602_170338ImgDisk1.tif")
    args = vars(ap.parse_args())


    #match=similarityImage(args["disks"], args["image"]);

    #Esto es para comprobar si lo que hemos obtenido se acerca a lo que deberiamos obtener
    similarityDataSet(args["disks"])
    '''
    (vectorEsp,vectorRe)= similarityDataSet(args["disks"]);
    target_names = ['AK-30', 'AMC-30', 'AMP-10',"Error"]
    comprobeResults(vectorRe,vectorEsp,target_names)
    '''