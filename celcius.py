import numpy as np
import tensorflow as tf


celsius = np.array([-40,-10, -9,26, 15, 100, 0, 21,30, 36, -5,21,8], dtype=float)
fahrenheit = np.array([-40,14, 15.8,78.8, 59, 212, 32, 69.8, 86,96.8, 23,69.8,46.4], dtype=float)

#keras es un framework que ahorra codigo al crear las capas

#capa = tf.keras.layers.Dense(units=1,input_shape=[1])
#modelo = tf.keras.Sequential([capa])
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2 ,salida])

#crear el modelo de la red neuronal
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)
#compilar el modelo con los parametros del optimizador 
print("Hola estoy comenzando el entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=2000, verbose=False)
print("... ya termine \n----- MODELO TERMINADO -----")

import matplotlib.pyplot as plt
plt.xlabel('# Epoca') 
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss']) # grafica para ver como se va aprendiendo
plt.show()

print("hagamos una prediccion")
prediccion = float(input())
resultado = modelo.predict([prediccion])

print("el resultado es " + str(resultado) + "°F")


#ver laestructura de la red 

print("ver variables internas del modelo")
print (oculta1.get_weights())
print (oculta2.get_weights())
print (salida.get_weights())

