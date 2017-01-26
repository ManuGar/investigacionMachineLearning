from detectanddescribe import DetectAndDescribe
from diskMatcher import DiskMatcher
import argparse
import cv2

from os import path
from glob import glob


matchIm= {}; #Lo necesitamos poner aqui para que se guarden de forma global todas las imagenes,
# de la otra forma como se hace referencia de forma recursiva la variable se reinicia cada vez que se mete en una carpeta nueva
#En esta diccionario guardamos las rutas de las imagenes que han obtenido alto % de coincidencia con la que hemos pasado y el % que tienen.

def similarityImage(carp,img):  # funcion principal

    maxItems=5 #Este es el numero maximo de elementos que vamos a permitir en el diccionario de los resultados
    di = glob(carp + "/" + "*")  # vemos todo lo que esta dentro del directorio que nos manden

    dad = DetectAndDescribe(cv2.FeatureDetector_create("FAST"),
                            cv2.DescriptorExtractor_create("FREAK"))

    queryImage = cv2.imread(img) #Leemos la imagen pasada como parametro
    (_, _, v) = cv2.split(cv2.cvtColor(queryImage, cv2.COLOR_BGR2HSV))

    (queryKps, queryDescs) = dad.describe(v);  # En v tenemos la imagen que estamos comparando en cada iteracion con el resto de imagenes de la carpeta

    for j in di:  # Recorremos el directorio, solo va a haber carpetas dentro por lo que j siempre sera una carpeta y asi evitamos comprobaciones de si es un archivo
        di2 = glob(j + "/" + "*") #Como dentro solo vamos a tener directorios con las clases de las imagenes, hacemos lo mismo que al principio
        # guardamamos la ruta de la carpeta en la que estamos en ese momento. En ese momento tenemo un array con todas las imagenes que estan dentro de esa carpeta

        cv = DiskMatcher(dad, di2, "BruteForce-Hamming") #Comparamos todas las imagenes dentro de la carpeta de la clase de ese momento
        # y le pasamos como parametro precisamente la carpeta de la clase actual (Ej: Ak-30) para hacer la comparacion
        #print("# of keypoints: {}".format(len(queryKps)))
        results = cv.search(queryKps, queryDescs); #Guardamos el vector con todos los resultados de la comparacion de la imagen con las de dentro de la carpeta

        if len(results) != 0:
                # loop over the results
            for (i, (score, diskPath)) in enumerate(results): #Recorremos el vector de resultados para aniadir solo las que tengan un % mas alto,
                # el limite de elementos lo ponemos en maxItems
                    if (score >= 0.65):
                        if (len(matchIm) < maxItems): #Solo queremos las 5 fotos que mas se parezcan a nuestra imagen
                            matchIm[diskPath] = score;
                            # print("{}. {:.2f} % {}".format(i + 1, score * 100, diskPath))
                            # cv2.imshow("Query", q)
                            # cv2.imshow("Match",r)
                            # cv2.waitKey(0)

                            # load the result image and show it
                            # result = cv2.imread(diskPath)
                            # cv2.imshow("Result", result)
                            # cv2.waitKey(0)
                        else:
                            minvalue = min(matchIm.values());
                            if (minvalue < score):
                                minkey = [key for key, value in matchIm.iteritems() if value == minvalue];  # Devuelve un vector con los valores minimos, puede ser que sean mas de uno
                                del matchIm[minkey[0]];  # borramos del diccionario el primero de los elementos que tenemos guardado en el vector
                                matchIm[diskPath] = score;

        #else:
            # print("I could not find a match for that disk!")
            # cv2.waitKey(0)
            # otherwise, matches were found

    '''
    Tenemos en matchIm las 5 imagenes que tienen mas parecido con la que le hemos pasado por parametro.
    Con ello ahora tenemos que decir de que tipo se
    predice que es mirando cual es la clase que mas se repite.

    Para eso crearemos un diccionario que tenga la clase como clave y el numero de apariciones como valor, asi devolveremos
    la clave que tenga mayor apariciones
    '''

    resultDic={}; #En esta variable guardaremos las rutas que han obtenido alto % de coincidencias y contaremos el numero de apariciones
    for pa, sc in matchIm.iteritems():
        route = pa.split("/");
        if (resultDic.has_key(route[1])):
            resultDic[route[1]]+=1;
        else: resultDic[route[1]]=1;

    #return matchIm
    #return resultDic;

    if (len(resultDic) != 0):
        maxx = max(resultDic.values());  # cogemos el maximo del diccionario de resultados
        for k in resultDic.items():
            if maxx == k[1]:
                return k[0]

'''
En este programa cogeremos lo que hace similarity y lo llevaremos a un ambito mas global. Ahora necesitamos saber el tipo
de todas las imagenes del data set. Tendremos que eliminar el caso de comparacion de la imagen que estamos usando en ese
momento consigo misma.
'''

def similarityDataSet(carp):
    di = glob(carp + "/" + "*")  # vemos todo lo que esta adentro del directorio que nos manden

    do = carp.split('/')

    for im in di:  # Recorremos el directorio

        '''
        Creo que lo tenemos que vaciar por que lo hemos puesto como una variable de la clase entonces se quedan los datos guardados desde la primera ejecucion
        y entonces lo que nos muestra no es lo que deberia ya que lo compara con los datos que tenia de la anterior ejecucion
        '''
        dir2 = glob(im + "/" + "*")
        for p in dir2:
            matchIm = {}
            print similarityImage(carp, p) + " prueba"



if __name__ == "__main__":  # Asi se ejecutan los scripts

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--disks", required=False, help="Path to the directory that contains our disks",
                    default="discPrueba")  # "images/a1.tif"
    ap.add_argument("-i", "--image", required=False, help="Path to the image that we want to compare",
                    default="subtract4.tif")
    args = vars(ap.parse_args())


    match=similarityImage(args["disks"], args["image"]);
    '''
    for path, sc in match.iteritems(): #Esto lo usabamos para imprimir todo lo que se guardaba en el diccionario
        print("{:.2f} % {}".format(sc * 100, path))
    '''

    '''
    for m in match.iteritems(): #Esto lo usabamos para imprimir cual es el tipo que mas veces aparece y el numero de apariciones
        print m;
    '''

    print match; #Imprimimos cual es el tipo mas repetido para la imagen
    similarityDataSet(args["disks"]);

#Es una copia del recorrido recursivo de carpetas
'''
def dirs(x, n):  # funcion principal
    di = glob(x + "/" + "*")  # vemos todo lo que esta adentro del directorio que nos manden
    tabs = "  " * n  # Algo de vista
    do = x.split('/')  # Esto es para mostrar el archivo(por culpa de glob)
    print tabs + do[-1] + " /"  # Mostramos la vista
    for i in di:  # Recorremos el directorio
        if path.isdir(i):  # Si es direcotorio
            dirs(i, n + 1)  # Empezamos la funcion recursiva
        elif path.isfile(i):  # Si es archivo
            do = i.split('/')  # gracias a glob :/
            print "|" + tabs + do[-1]  # Lo mostramos


if __name__ == "__main__":  # Asi se ejecutan los scripts
    dirs(".", 2)  # Iniciamos el script buscando en el directorio actual


'''

