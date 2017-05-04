# -*- coding: utf-8 -*-
from detectanddescribe import DetectAndDescribe
from diskMatcher import DiskMatcher
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, KFold
from time import time
from multiprocessing import Manager, Process
import multiprocessing as mp
import argparse
import cv2
import pandas as pd
import os
import itertools as it
import traceback



'''
Como los procesos por defecto no pueden lanzar excepciones, se ha heredado de la clase Process para 
implementar de alguna forma la forma de saber que ha habido una excepción durante la ejecución del proceso.
Con esta clase, ya podremos saberlo y seguir lanzando la excepción hasta donde nosotros queremos.
'''
class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def similarityImage(imageVector,img, featureDetector, descriptorExtractor, diskMatcher):
    # En esta diccionario guardamos las rutas de las imagenes que han obtenido alto % de coincidencia con la que hemos pasado y el % que tienen.
    matchIm = {}
    maxItems=5 #Este es el numero maximo de elementos que vamos a permitir en el diccionario de los resultados
    queryImage = cv2.imread(img)  # Leemos la imagen pasada como parametro
    dad = DetectAndDescribe(eval(featureDetector),
                                eval(descriptorExtractor))  # Creamos un objeto DetectAndDescribe (creado por Jónathan) al que le pasamos los objetos
        # que detectan los puntos clave de la imagen y recoge los descriptores a partir de esos puntos clave

    (_, _, v) = cv2.split(cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV))
    (queryKps, queryDescs) = dad.describe(v);  # En v tenemos la imagen que estamos comparando en cada iteracion con el resto de imagenes de la carpeta
    cv = DiskMatcher(dad, imageVector, diskMatcher) #Comparamos todas las imagenes dentro del vector de imagenes que le pasamos como parametro
    results = cv.search(queryKps, queryDescs); #Guardamos el vector con todos los resultados de la comparacion de la imagen con las de dentro de la carpeta

    if len(results) != 0:
        for (i, (score, diskPath)) in enumerate(results): #Recorremos el vector de resultados para aniadir solo las que tengan un % mas alto,
             # el limite de elementos lo ponemos en maxItems
            if (score >= 0.65 ):
                if (len(matchIm) < maxItems): #Solo queremos las maxItems(5) fotos que mas se parezcan a nuestra imagen
                    matchIm[diskPath] = score;
                else:
                    minvalue = min(matchIm.values());
                    if (minvalue < score):
                        minkey = [key for key, value in matchIm.iteritems() if value == minvalue];  # Devuelve un vector con los valores minimos, puede ser que sean mas de uno
                        del matchIm[minkey[0]];  # borramos del diccionario el primero de los elementos que tenemos guardado en el vector
                        matchIm[diskPath] = score;

    '''
    Tenemos en matchIm las 5 imágenes que tienen más parecido con la que le hemos pasado por parámetro.
    Con ello ahora tenemos que decir de que tipo se predice que es mirando cual es la clase que mas se repite.
    Para eso crearemos un diccionario que tenga la clase como clave y el número de apariciones como valor,
    así devolveremos la clave que tenga mayor apariciones
    '''
    resultDic={}; #En esta variable guardaremos las rutas que han obtenido alto % de coincidencias y contaremos el número de apariciones
    #para luego devolver la clase que tenga mayor número de apariciones
    for pa, sc in matchIm.iteritems():
        route = pa.split("/");
        if (resultDic.has_key(route[1])):
            resultDic[route[1]]+=1;
        else: resultDic[route[1]]=1;

    if (len(resultDic) != 0):
        maxx = max(resultDic.values());  # cogemos el máximo del diccionario de resultados
        for k in resultDic.items():
            if maxx == k[1]:
                return k[0]
    else: return "Error"

'''
En este programa cogeremos lo que hace similarity y lo llevaremos a un ambito mas global. Ahora necesitamos saber el tipo
de todas las imagenes del data set. Tendremos que eliminar el caso de comparacion de la imagen que estamos usando en ese
momento consigo misma.
'''
def similarityDataSet(carp, featureDetector, descriptorExtractor, diskMatcher):
    n_splits=10
    di = glob(carp + "/" + "*")  # vemos todo lo que está dentro del directorio que nos manden
    images=[]
    target_names = []
    for j in di:
        target_names.append(j.split("/")[-1])# 'AK-30', 'AMC-30', 'AMP-10', "Error"
        images+=(glob(j + "/" + "*"))

    target_names.append("Error")
    '''
    Con esto lo que hacemos es generar un vector con los indices que queremos en entrenamiento y las que queremos 
    en test. Luego creamos un vector de las imagenes de entrenamiento a partir de los indices y recorremos el vector
    de las de test para hacer la comparacion.
    '''
    kf= KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores=[]
    pool = [] #Pool(num_processes)
    splits =[] #En esta variable guardamos las variables de todos los pasos para que luego map pueda ejecutarlas de forma paralela
    manager = Manager() #Objeto que nos permite crear una cola con para compartir los datos entre los procesos
    queue = manager.Queue() #Creamos una cola donde guardaremos los accuracy que vayan dejando los procesos que ya se hayan ejecutado
    for train_index, test_index in kf.split(images):
        varsSet = (train_index, test_index, images, target_names,
                   featureDetector, descriptorExtractor, diskMatcher,queue)
        splits.append(varsSet)
        pool.append(Process(target=calculate_split, args=(varsSet,)))

    for p in pool:
        p.start()

    while pool:
        for p in pool:
            if not p.is_alive():
                p.join
                if p.exception:
                    raise Exception
                pool.remove(p)
                del(p)

    queue.put('DONE') #Para decirle al bucle cuando acabar y salir
    while True:
        result = queue.get()
        if (result == 'DONE'):
            break
        else:
            accuracy_scores.append(result)
    createCSV("results.csv",accuracy_scores, featureDetector, descriptorExtractor, diskMatcher)

def calculate_split (x):
    '''Metodo para poder hacer la paralelizacion del programa. Esta funcion se ejecuta por cada split de kf (KFold) y calcula todas las comparaciones
    y los datos referentes a un split, como puede ser la accuracy y nos muestra los datos de hacer la comparacion de las imagenes que tenemos en este split
    '''
    (train_index, test_index, images, target_names,
     featureDetector, descriptorExtractor, diskMatcher, queue) = x
    vectorExpectedType = []
    vectorRealType = []
    train_images = []
    for i in train_index:
        train_images.append(images[i])
    for index in test_index:
        print images[index] #Lo imprimimos para saber en que imagen esta el programa en un momento determinado
        vectorExpectedType.append(
            similarityImage(train_images, images[index], featureDetector, descriptorExtractor, diskMatcher))
        vectorRealType.append(images[index].split("/")[1])

    printResults(vectorRealType, vectorExpectedType, target_names)
    queue.put(accuracy_score(vectorRealType, vectorExpectedType))
    return (accuracy_score(vectorRealType, vectorExpectedType))

def printResults(y_true, y_pred,target_names): #método para mostrar los datos después de haber hecho la comparación de las imágenes
    print "These are the predicted type of the images"
    print y_pred
    print "These are the real type of the images"
    print y_true
    print classification_report(y_true, y_pred, target_names=target_names)
    print confusion_matrix(y_true, y_pred, labels=target_names)
    print "\n",accuracy_score(y_true,y_pred),"\n"
    '''
    Con esto conseguimos guardar todos los datos en una estructura de pandas y una vez tenemos la estructura lo que hacemos es crear
    un archivo csv si no estaba creado y si lo estaba añadirle los datos
    '''

def createCSV(name,accuracy_scores, featureDetector, descriptorExtractor, diskMatcher):
    ''' Acumular el accurancy_score en un vector y lo pasamos al csv al final para hacer solo una tarea de entrada/salida
        Los datos se guardan en results y se añaden a un dataframe que se creamos con la orientación que necesitamos para
        ver mejor todos los datos de las pruebas
    '''
    results = [(str(featureDetector).split(".")[-1] + "-" + str(descriptorExtractor).split(".")[-1] + "-" + diskMatcher , accuracy_scores)]
    df = pd.DataFrame.from_items(results,orient='index',columns=range(0,len(accuracy_scores)))
    if not os.path.isfile(name):
        df.to_csv(name, encoding='utf-8')
    else:  # else it exists so append without writing the header
        df.to_csv(name, mode='a', header=False,encoding='utf-8')

def createCSVTime(name,totalTime, featureDetector, descriptorExtractor, diskMatcher):
    ''' Generamos otro CSV pero para guardar en este caso los tiempos de ejecución de cada uno de los métodos.
        Guardamos únicamente el tiempo que le ha costado en total a cada método, es decir, que se guardará el tiempo que
        le haya costado ejecutar todas las iteraciones que tengamos configuradas para el conjunto de estos 3 parámetros
    '''
    results = [(str(featureDetector).split(".")[-1] + "-" + str(descriptorExtractor).split(".")[-1] + "-" + diskMatcher, [totalTime])] #revisarlo que da problemas
    df = pd.DataFrame.from_items(results,orient='index',columns=["Time(s)"])
    if not os.path.isfile(name):
        df.to_csv(name,encoding='utf-8')
    else:  # else it exists so append without writing the header
        df.to_csv(name, mode='a', header=False,encoding='utf-8')

def executeCombinations(folder):

    featureDetectors = [ "cv2.xfeatures2d.StarDetector_create()", "cv2.ORB_create()", "cv2.AKAZE_create()",
                        "cv2.FastFeatureDetector_create()","cv2.MSER_create()","cv2.xfeatures2d.SIFT_create()",
                        "cv2.xfeatures2d.SURF_create()"]

    descriptorExtractors = ["cv2.xfeatures2d.SIFT_create()", "cv2.ORB_create()", "cv2.BRISK_create()", "cv2.AKAZE_create()",
        "cv2.xfeatures2d.BriefDescriptorExtractor_create()", "cv2.xfeatures2d.FREAK_create()","cv2.xfeatures2d.SURF_create()"
        ]

    diskMatchers = ["BruteForce-Hamming", "BruteForce", "BruteForce-L1", "BruteForce-Hamming(2)", "FlannBased"]

    '''
        cv2.xfeatures2d.DAISY_create() #No esta???
        cv2.xfeatures2d.MSDDetector_create() #También debería estar 
    '''

    '''
    featureDetector = 'cv2.ORB_create()'
    descriptorExtractor = 'cv2.ORB_create()'
    diskMatcher = 'BruteForce-L1'
    init_time = time()
    similarityDataSet(args["disks"], featureDetector, descriptorExtractor, diskMatcher)
    total_time =time() - init_time
    print "Execuction time (s): ", total_time
    createCSVTime(total_time,featureDetector,descriptorExtractor,diskMatcher)
    '''

    for (featureDetectors, descriptorExtractors, diskMatchers) in it.product(featureDetectors, descriptorExtractors, diskMatchers):
        print(featureDetectors, descriptorExtractors, diskMatchers)
        start_time = time()
        try:
            similarityDataSet(folder, featureDetectors, descriptorExtractors, diskMatchers)
            total_time = time() - start_time
            print "Execution time(s): ", total_time
            createCSVTime("times.csv",total_time, featureDetectors, descriptorExtractors, diskMatchers)
        except Exception as inst:
            print(type(inst))  # la instancia de excepción
            print(inst.args)  # argumentos guardados en .args
            print(str(inst)+ "\n")
            createCSV("resultsError.csv", ["Combinación no compatible"], featureDetectors, descriptorExtractors, diskMatchers)
            #createCSVTime("timesError.csv", "Combinación no compatible", featureDetectors, descriptorExtractors, diskMatchers)

if __name__ == "__main__":  # Así se ejecutan los scripts
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--disks", required=False, help="Path to the directory that contains our disks",
                    default="datasetPrueba")
    ap.add_argument("-i", "--image", required=False, help="Path of the image we want to compare",
                    default="discPrueba/AK-30/rot33.tiff")
    args = vars(ap.parse_args())
    executeCombinations(args["disks"])
