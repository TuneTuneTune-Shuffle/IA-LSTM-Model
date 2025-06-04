import os
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt

#from tensorflow_datasets import testing
from tensorflow import keras
#import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

from davids_LabelMaker import getGenre

def save_model(model, path):
    """Guarda un modelo de TensorFlow en el path especificado."""
    model.save(path)
    print(f"Modelo guardado en: {path}")

def load_model(path):
    """Carga un modelo de TensorFlow desde el path especificado."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en: {path}")
    model = tf.keras.models.load_model(path)
    print(f"Modelo cargado desde: {path}")
    return model

def save_mfcc_dataset(dataset, labels, path):
    """Guarda el dataset MFCC y sus etiquetas en archivos .npy"""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "mfcc_data.npy"), dataset)
    np.save(os.path.join(path, "mfcc_labels.npy"), labels)
    print(f"MFCC y etiquetas guardadas en: {path}")

def load_mfcc_dataset(path):
    """Carga un dataset MFCC y sus etiquetas desde archivos .npy"""
    X = np.load(os.path.join(path, "mfcc_data.npy"))
    y = np.load(os.path.join(path, "mfcc_labels.npy"))
    return X, y

def predict_genre(model, mfcc, index_to_label):
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)[0]
    return index_to_label[predicted_index]

#Funcion que agarra un numero de audios para luego convertirlos a MFCC (Mel-frequency cepstral coefficient)
def audioMFCC(audio):
    # print(f"Procesando audio: {audio}")
    # print(f"Duración del audio: {librosa.get_duration(filename=audio)} segundos")
    # print("Type de audio:", type(audio))
    audio_series , sample_rate = librosa.load(audio,duration = 30.0,res_type="soxr_hq")
    mfcc = librosa.feature.mfcc(y=audio_series, sr=sample_rate, n_mfcc=40)
    #np.ndarray [shape=(…, n_mfcc, t)]
    return mfcc,audio_series

#Funcion que recorre las carpetas y devuelve un dataset con su labelset
def readDataSet(ds_path,num=100):
        labels_pd = getGenre()
        datasetMFCC = []
        labelsMFCC = []
        num_items = 0

        #Ordenamos nuestras carpetas
        chunck = sorted(os.listdir(ds_path))
        
        for _, file in enumerate(chunck):
            if num_items > num:
                    break
            # print(f"file: {file}")
            file_path = os.path.join(ds_path,file)

            if not os.path.isdir(file_path):
                continue  # Ignorar archivos sueltos

            print(f"Procesando carpeta: {file}")
            for song  in os.listdir(file_path):
                if num_items > num:
                    break
                # print(f"song: {int(os.path.splitext(song)[0])}")
                label = labels_pd[labels_pd['track_id'] == int(os.path.splitext(song)[0])]['track_genre_top'].iloc[0]
                # print(f"song : {song}")
                # print(f"label: {label}")
                if song.endswith(".mp3"):
                    song_path = os.path.join(file_path, song)
                try:
                    y = file_path
                    mfcc,audio_series = audioMFCC(song_path)
                    if len(audio_series) < 2048:
                        print(f"Audio muy corto, saltando: {y}")
                        continue
                    datasetMFCC.append(mfcc)
                    labelsMFCC.append(label)
                    num_items+=1
                except Exception as e:
                    print(f"Error procesando {song_path}: {e}")
                    print(f"Error procesando {y}: {e}")
                   

        return datasetMFCC, labelsMFCC

#Padding
def pad_mfcc(mfcc, max_len=100):
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        return mfcc[:, :max_len]

            
# Crear modelo
def genreClassifier(input_shape,num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

#Grafica
def plot_history(history):
    """
    Plots the training and validation accuracy and loss from a Keras History object.

    Args:
        history: The History object returned by model.fit().
    """
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.ylabel('Precisión')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True) # Add a grid for better readability

    # Subplot 2: Loss
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.ylabel('Pérdida')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True) # Add a grid for better readability

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def main():
    print("Dispositivos disponibles:")
    print(tf.config.list_physical_devices('GPU'))

    #Creamos el path al dataset
    ds_path = '/home/assessment1/Documents/JpHusbandWorkSpace/IA-LSTM-Model/Dataset/fma_large/'

    #Creamos nuestro dataSet y el labelSet
    dataSet , labelSet = readDataSet(ds_path)

    # Crear el mapeo de etiquetas: género -> número
    unique_labels = sorted(set(labelSet))
    print(f"Etiquetas únicas encontradas: {unique_labels}")
    
    #Recorremos las etiquetas y creamos un diccionario de mapeo el cual inidca conocer la canntidad de géneros
    # y la cantidad de MFCC que tenemos
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"Diccionario de etiquetas: {label_to_index}")

    # index_to_label = {idx: label for label, idx in label_to_index.items()}  # opcional, para decodificar lueg
    # #Aplicar padding a todos
    #X = np.array([pad_mfcc(mfcc) for mfcc in dataSet])#, dtype=np.float32)
    X = np.array([pad_mfcc(mfcc) for mfcc in dataSet])
    print(X)
   
    y = np.array([label_to_index[label] for label in labelSet])
    print(y)

    # Guardamos el dataset original sin padding (opcional)
    save_mfcc_dataset(X, y, "./LSTM-SaveData&&Model")

    num_classes = len(label_to_index)  

    #Imprimimos el primer mfcc del dataset
    print("Cantidad de MFCC:", len(dataSet))
    print("Cantidad de etiquetas:", len(labelSet))
    print("Tamaño de un MFCC:", dataSet[0].shape)
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(dataSet[0], x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.show()

    # División 80% entrenamiento, 20% validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Cantidad de géneros:", num_classes)
    print("Cantidad de MFCC:", len(X))
    print("Cantidad de etiquetas:", len(y))

    print("Cantidad de MFCC para entrenamiento:", len(X_train))
    print("Cantidad de MFCC para validación:", len(X_val))
    BATCH_SIZE = 32

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    input_shape = X_train.shape[1:]  # (1300, 40)

    model = genreClassifier(input_shape,num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    checkpoint_path = "./LSTM-SaveData&&Model/best_model.h5"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=30)

    test_loss, test_acc = model.evaluate(val_ds)
    print('Test accuracy:', test_acc)

    model.save("./LSTM-SaveData&&Model/genre_classifier_model.h5")

    plot_history(history)

if __name__ == "__main__":
    main()