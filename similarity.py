from detectanddescribe import DetectAndDescribe
from diskMatcher import DiskMatcher
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split, KFold
from time import time
from multiprocessing import Pool
import argparse
import cv2
import pandas as pd
import os

#Lo necesitamos poner aqui para que se guarden de forma global todas las imagenes,
# de la otra forma como se hace referencia de forma recursiva la variable se reinicia cada vez que se mete en una carpeta nueva
#En esta diccionario guardamos las rutas de las imagenes que han obtenido alto % de coincidencia con la que hemos pasado y el % que tienen.
def similarityImage(imageVector,img, featureDetector, descriptorExtractor, diskMatcher):  # funcion principal
    matchIm = {}
    maxItems=5 #Este es el numero maximo de elementos que vamos a permitir en el diccionario de los resultados

    #cv2.xfeatures2d.
    #cv2.xfeatures2d.SIFT_create()
    #cv2.FeatureDetector_create(featureDetector)
    #cv2.FastFeatureDetector()
    #

    dad = DetectAndDescribe(featureDetector, #Esto habra que cambiarlo por el metodo de la version nueva
                            descriptorExtractor) #Habra que cambiarlo por el metodo de la version nueva

    queryImage = cv2.imread(img) #Leemos la imagen pasada como parametro
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
def similarityDataSet(carp, featureDetector, descriptorExtractor, diskMatcher):
    tiempo_total = 0
    n_splits=10
    num_processes = 10
    di = glob(carp + "/" + "*")  # vemos todo lo que esta adentro del directorio que nos manden
    images=[]
    target_names = []
    for j in di:
        target_names.append(j.split("/")[-1])# 'AK-30', 'AMC-30', 'AMP-10', "Error"
        images+=(glob(j + "/" + "*"))

    target_names.append("Error")

    '''
    Para hacer la prueba 10 veces lo que podemos es hacer es crear un bucle que vaya de uno a diez o mediante
    la funcion Kfold de sklearn

    Con Kfold es un poco mas largo pero parece un poco mas rapido tambien. Con esto lo que hacemos es generar
    un vector con los indices que queremos en entrenamiento y las que queremos en test. Luego creamos un vector
    de las imagenes de entrenamiento a partir de los indices y recorremos el vector de las de test para hacer
    la comparacion.
    '''
    kf= KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores=[]

    pool = Pool(processes = num_processes)

    for train_index, test_index in kf.split(images):
        conjunto_variables = (train_index, test_index, images, target_names, tiempo_total,
                              accuracy_scores, featureDetector, descriptorExtractor, diskMatcher)
        tiempo_y_accuracy=pool.map(calc_split,(conjunto_variables,))
        (tiempo,accuracy) = tiempo_y_accuracy[0]
        tiempo_total +=tiempo
        accuracy_scores.append(accuracy)

    createCSV(accuracy_scores, featureDetector, descriptorExtractor, diskMatcher)

    '''
    for i in range(0, n_splits):
        start_time = time()
        vectorExpectedType = []
        vectorRealType = []
        (X_train, images_test, _, _) = train_test_split(images, np.zeros(len(images)), test_size=0.25, random_state=i)
        for j in images_test:
            vectorExpectedType.append(similarityImage(X_train, j, featureDetector, descriptorExtractor, diskMatcher))
            vectorRealType.append(j.split("/")[1])

        comprobeResults(vectorRealType, vectorExpectedType, target_names)
        tiempo_iteracion = time() - start_time
        print "El tiempo de esta iteracion es: " + str(tiempo_iteracion)
        tiempo_total+=tiempo_iteracion
        print "El tiempo total de la ejecucion del programa ha sido: " + str(tiempo_total)
    '''
    print "Mean iterations execution time has been: " + str(tiempo_total/n_splits)
    print "Total time has been: " + str(tiempo_total)

def calc_split (x):
    '''Metodo para poder hacer la paralelizacion del programa. Esta funcion se ejecuta por cada split de kf (KFold) y calcula todas las comparaciones
    y los datos referentes a un split, como puede ser la accuracy y nos muestra los datos de hacer la comparacion de las imagenes que tenemos en este split
    '''
    (train_index, test_index, images, target_names, tiempo_total,
     accuracy_scores, featureDetector, descriptorExtractor, diskMatcher) = x
    start_time = time()
    vectorExpectedType = []
    vectorRealType = []
    train_images = []
    for i in train_index:
        train_images.append(images[i])

    for index in test_index:
        #print images[index] #Es para saber con que imagen esta trabajando en cada momento por si da algun error
        vectorExpectedType.append(
            similarityImage(train_images, images[index], featureDetector, descriptorExtractor, diskMatcher))
        vectorRealType.append(images[index].split("/")[1])

    comprobeResults(vectorRealType, vectorExpectedType, target_names)
    tiempo_iteracion = time() - start_time
    #tiempo_total += tiempo_iteracion
    #accuracy_scores.append(accuracy_score(vectorRealType, vectorExpectedType))
    tiempo_y_accuracy=(tiempo_iteracion,accuracy_score(vectorRealType, vectorExpectedType))
    return tiempo_y_accuracy

def comprobeResults(y_true, y_pred,target_names): #metodo para mostrar los datos despues de haber hecho la comparacion de las imagenes
    print "These are the predicted type of the images"
    print y_pred
    print("\n")
    print "These are the real type of the images"
    print y_true
    classi_rep=classification_report(y_true, y_pred, target_names=target_names)
    print(classi_rep)
    conf_mat=confusion_matrix(y_true, y_pred, labels=target_names)
    print conf_mat
    print "\n"
    print accuracy_score(y_true,y_pred)
    '''
    Con esto conseguimos guardar todos los datos en una estructura de pandas y una vez tenemos la estructura lo que hacemos es crear
    un archivo csv si no estaba creado y si lo estaba aniadirle los datos
    '''

def createCSV(accuracy_scores, featureDetector, descriptorExtractor, diskMatcher):
    ''' Acumular el accurancy_score en un vector y lo pasamos al csv al final para hacer solo una tarea de entrada/salida
        Los datos se guardan en results y se aniaden a un dataframe que se creamos con la orientacion que necesitamos para
        ver mejor todos los datos de las pruebas
    '''
    results = [(featureDetector + "-" + descriptorExtractor + "-" + diskMatcher , accuracy_scores)]
    df = pd.DataFrame.from_items(results,orient='index',columns=range(0,len(accuracy_scores)))
    if not os.path.isfile('results.csv'):
        df.to_csv('results.csv')
    else:  # else it exists so append without writing the header
        df.to_csv('results.csv', mode='a', header=False)

if __name__ == "__main__":  # Asi se ejecutan los scripts
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--disks", required=False, help="Path to the directory that contains our disks",
                    default="discPrueba")

    args = vars(ap.parse_args())
    featureDetector=cv2.ORB_create#cv2.xfeatures2d.SIFT_create()   "FAST"
    descriptorExtractor=cv2.ORB_create #cv2.xfeatures2d.FREAK_create() "FREAK"
    diskMatcher= "BruteForce-Hamming"
    similarityDataSet(args["disks"], featureDetector, descriptorExtractor, diskMatcher)
