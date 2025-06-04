#Script que contiene las funciones princiaples para ser llamadas}


#Funcion para guardar una funcion en wav y convertirla en MFCC en tiempo real
def getSong():
    #Aqui hacer logica con microfono

#Funcion de analizar cancion por genero

def songGenreAnalyzer(model,song):
    #Usamos el modelo para devolver el genero de una cancion
    return model(song)

