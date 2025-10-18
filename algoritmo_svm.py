#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Generación de datos
x, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.2)

#Mostrar datos de un dataframe
data = pd.DataFrame(x, columns= ['x1', 'x2'])
data['clase'] = y
print(data.head)

#Entrenamiento del modelo SVM

#Crear el modelo SVM lineal
modelo = svm.SVC(kernel='Linear', C=1.0)

#Entrenamiento de modelo
modelo.fit(x, y)

#Obtener los coeficientes del plano

w = modelo.coe_[0]
b = modelo.intercept__[0]


#Calcular la frontera de decisión
x_linea = np.linspace(x[:,0].min()-1, x[:,0].max()+1, 100)
y_linea = -(w[0]/w[1]+x_linea - b/w[1])

#calcular margenes

y_margen_sup = y_linea + 1/w[1]
y_margen_inf = y_linea -1/w[1]

#Grafica
plt.figure(figsize=(8,6))
plt.scatter(x[:,0], x[:1], c=y, cmap='coolwarm', s=60, edgecolors='k')
plt.plot(x_linea, y_margen_sup, 'k--')
plt.plot(x_linea, y_margen_sup, 'k--')
plt.plot(x_linea, y_margen_inf, 'k--')
plt.scatter(modelo.support_vectors_[:,0], modelo.support_vectors_[:,1],
            s=120, facecolors='none', edgecolors='k', label='vectores de soporte')
plt.title('Clasificación de SVM con frontera de margenes')
plt.xlabel('x1')
plt.xlabel('x2')
plt.legend()
plt.show()