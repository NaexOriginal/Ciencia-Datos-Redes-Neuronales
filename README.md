# Ciencia-Datos-Redes-Neuronales
___
## La primera red neuronal con TensorFlow + Keras
> Para este proyecto es recomendado hacerlo en un entorno de nube como **Google Colab** o similar. En caso de tener buen nivel de cómputo, se puede realizar desde el equipo personal en la sección de notas de un editor de código como **VS Code**.

**Dataset:** Iris (150 muestras, 4 features, 3 clases)  
**Objetivo:** Clasificar especies de flores con una red neuronal de 2 capas ocultas.
___
## Bloque 1: Importar Librerías
Para el realizar nuestro modelo usaremos librerías que ya conocemos como lo son **Numpy**, **Pandas** y **MatplotLib**, pero para poder realizar correctamente este modelo usaremos nuevas librerías que nos facilitar realizar nuestra primer red Neuronal

| Librería     | ¿Qué hace?                                                                                                                                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tensorflow` | El motor de inteligencia artificial desarrollado por Google. Ejecuta todos los cálculos de la red neuronal de manera básica.                                                                                                 |
| `keras`      | Un complemento conocido como la interfaz amigable de TensorFlow. En lugar de escribir matemática compleja, Keras nos deja construir redes en pocas líneas sin la necesidad de aprender todo el concepto matemático de golpe. |
| `sklearn`    | Librería de machine learning clásico (No usado en entornos reales). Aquí la usamos solo para cargar un dataset y preparar los datos.                                                                                         |

```python
# Las mismas librerías que conocen + TensorFlow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow versión:", tf.__version__)
```

## Bloque 2: Cargar y Explorar los Datos
Siguiendo de manera similar al proceso de CRISP-DM que ya conocemos tenemos que: **antes de modelar, entendemos los datos.**

`load_iris()` descarga directamente el dataset Iris, uno de los más usados para aprender machine learning. Contiene medidas de 150 flores divididas en 3 especies. Lo convertimos a un DataFrame de Pandas para poder explorarlo
```python
# Cargar dataset, primero entendemos los datos
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target  # 0=setosa, 1=versicolor, 2=virginica

print("Figura de X:", X.shape)
print("Clases:", iris.target_names)
X
```

> `iris.data` contiene las medidas de las flores (las entradas). `iris.target` contiene la especie de cada flor expresada como número (la salida que queremos predecir). `X.shape` devuelve `(150, 4)`: 150 flores, 4 medidas cada una.
___
A continuación dividimos los datos y los escalamos. Estas dos operaciones son **obligatorias** antes de entrenar cualquier red neuronal.

**`train_test_split`** divide el dataset en dos partes: una para que la red aprenda (`X_train`) y otra para evaluar qué tan bien aprendió (`X_test`). El parámetro `test_size=0.2` reserva el 20% para prueba. `random_state=42` garantiza que la división sea siempre la misma, esto es importante para que los resultados sean reproducibles y fácilmente entendibles.

Por otro lado tenemos que usar el método **`StandardScaler`** , en otras palabra escala los datos para que todas las variables estén en la misma magnitud (media = 0, desviación = 1). Esto es crítico: si una columna va de 0 a 1 y otra de 0 a 1000, la red le prestará atención desproporcionada a la segunda y aprenderá mal.
```python
# Dividir en train/test — igual que siempre
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar las redes neuronales
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)
```

## Bloque 3 — Construir la red neuronal
Aquí definimos la **arquitectura**: cuántas capas tiene la red, cuántas neuronas por capa y qué función de activación usa cada una.

`keras.Sequential` construye la red capa por capa, en orden. Cada `layers.Dense` es una capa **completamente conectada**: todas las neuronas de esa capa están conectadas con todas las de la capa anterior.

| Capa     | Neuronas | Activación | ¿Por qué?                                   |
| -------- | -------- | ---------- | ------------------------------------------- |
| Entrada  | 4        | —          | Una por cada columnas (feature) del dataset |
| Oculta 1 | 16       | ReLU       | Aprende patrones en los datos               |
| Oculta 2 | 8        | ReLU       | Refina lo que aprendió la capa anterior     |
| Salida   | 3        | Softmax    | Una probabilidad por cada especie de flor   |

```python
model = keras.Sequential([
    # Capa de entrada implícita: 4 features
    layers.Dense(16, activation='relu', input_shape=(4,)),
    # Capa oculta 2
    layers.Dense(8,  activation='relu'),
    # Capa de salida: 3 clases con Softmax
    layers.Dense(3,  activation='softmax')
])

model.summary()
```

## Bloque 4 — Compilar y entrenar
**Compilar** es configurar las reglas de aprendizaje antes de arrancar el entrenamiento.

- `optimizer='adam'` es el algoritmo que ajusta los pesos después de cada error.
- `loss='sparse_categorical_crossentropy'` es la función que mide qué tan equivocada estuvo la red. Cuanto más bajo, mejor. Se usa esta variante específica cuando las clases son números enteros (0, 1, 2).
- `metrics=['accuracy']` indica que queremos ver el porcentaje de aciertos durante el entrenamiento.
```python
model.compile(
    optimizer='adam',                        # algoritmo que ajusta los pesos
    loss='sparse_categorical_crossentropy',  # función de error
    metrics=['accuracy']                     # métrica que queremos ver
)
```
___
**Entrenar** es el momento en que la red ajusta sus parámetros mirando los datos repetidamente.

- `epochs=50` significa que la red va a ver **todo el dataset 50 veces**. En cada pasada ajusta sus pesos para cometer menos errores.
- `validation_split=0.2` reserva el 20% del training para validar en tiempo real, así podemos ver si la red está aprendiendo bien o memorizando.
- `verbose=1` muestra el progreso en consola. Observa cómo el `loss` baja y el `accuracy` sube con cada época.

```python
history = model.fit(
    X_train, y_train,
    epochs=50,               # veces que ve todo el dataset
    validation_split=0.2,    # 20% para validar mientras entrena
    verbose=1
)
```

## Bloque 5 — Visualizar el aprendizaje
Graficamos las **curvas de aprendizaje** usando Matplotlib: cómo cambiaron la pérdida y la precisión a lo largo de las 50 épocas, tanto en entrenamiento como en validación.
```shell
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Gráfica 1: pérdida
ax1.plot(history.history['loss'],     label='Entrenamiento')
ax1.plot(history.history['val_loss'], label='Validación')
ax1.set_title('Pérdida por época')
ax1.set_xlabel('Época')
ax1.legend()

# Gráfica 2: precisión
ax2.plot(history.history['accuracy'],     label='Entrenamiento')
ax2.plot(history.history['val_accuracy'], label='Validación')
ax2.set_title('Precisión por época')
ax2.set_xlabel('Época')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Bloque 6 — Evaluar el modelo
Evaluamos la red con datos que **nunca vio durante el entrenamiento**: el conjunto de prueba. Este es la prueba para rectificar que todo funcione, si la red aprendió bien, debería funcionar igual de bien con datos nuevos, es decir, podrá predecir cual es la flor correcta.

Además se usa `model.evaluate` el cual devuelve el error y la precisión final. De igual manera que `model.predict` el cual recibe una muestra y devuelve las probabilidades para cada clase, al final con `argmax()` establece cual es la flor que considera que es correcta
```python
# Evaluación en datos que nunca vio
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en test: {accuracy:.2%}")

# Hacer una predicción manual
muestra = X_test[0].reshape(1, -1)
prediccion = model.predict(muestra)

print(f"\nProbabilidades: {prediccion[0].round(3)}")
print(f"Clase predicha: {iris.target_names[prediccion[0].argmax()]}")
print(f"Clase real:     {iris.target_names[y_test[0]]}")
```
