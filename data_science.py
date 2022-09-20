#-----------------------------------------------------
# Julio Cesar Galvez Espinosa
#-----------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns  #Generador de graficas
import matplotlib.pyplot as plt #Muestra de graficas
from scipy import stats #Calculo coeficiente de correlación entre medidas
from statsmodels.stats.weightstats import ztest #Evaluación de hipotesis
import statsmodels.api as sm #Generador formula para Regresión Logística.
import os

# 1) - Extracción del path de la data de flores.
irisfile_path = os.getcwd() + '/data_sets/iris_data.csv'

#-------------------------------------------------------------------------
# 2) - Carga del archivo .csv con la data de flores (DataFrame de Pandas)
#-------------------------------------------------------------------------
iris_original = pd.read_csv(irisfile_path)
#print(type(iris_original))

#----------------------------------------------------
# 3) - Naturaleza del DataFrame
#----------------------------------------------------
print('\n')
print('#----------------------------------------------------')
print('#------Naturaleza del DataFrame----------------------')
print('#----------------------------------------------------')
print(iris_original.info())

#----------------------------------------------------
# 4) - Algunos datos del DataFrame
#----------------------------------------------------
print('\n')
print('#----------------------------------------------------')
print('#------Algunos datos del DataFrame----------------------')
print('#----------------------------------------------------')
print(iris_original.head())


#----------------------------------------------------
# 5) - Visualización relación entre variables
#----------------------------------------------------
print('\n')
print('#----------------------------------------------------')
print('#------Visualización relación entre variables--------')
print('#----------------------------------------------------')
visualizar = input("Desea visualizar la relación entre variables? (S/N) ")
if visualizar == 'S' or visualizar == 's':
    sns.pairplot(iris_original, hue='label')
    plt.show()


#---------------------------------------------------------------
# 6) - Calculo de media y desviación estándar de las variables
#--------------------------------------------------------------
print('\n')
print('#---------------------------------------------------------')
print('#-Calculo de media y desviación estándar de las variables-')
print('#---------------------------------------------------------')
print(iris_original.groupby('label').agg([np.mean, np.std]).T)


#---------------------------------------------------------------
# 7) - Creación de un nuevo DataFrame
#--------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Creación de un nuevo DataFrame:')
print('#-1)-Solo la información de especies versicolor y virginica')
print('#-2)-Se incluye una característica adicional "es_virginica: 1=virginica/0=versicolor')
print('#-----------------------------------------------------------------------------------')
iris_df = iris_original.copy() #Realiza la copia del Dataframe inicial
print("Columnas del dataframe original: {}".format(iris_df.columns))

#definición de un diccionario con el label origen vs label destino
dicc= { 'sepal length (cm)': 'longitud_sépalo' ,
        'sepal width (cm)' : 'ancho_sépalo'    ,
        'petal length (cm)': 'longitud_pétalo' ,
        'petal width (cm)' : 'ancho_pétalo'    ,
        'label'            : 'especie'
}

#Renombramiento de columnas
iris_df = iris_df.rename(columns=dicc)

#Dataframe sin la data de 'setosa'
iris_df = iris_df[iris_df['especie'] != 'setosa']

#Se agregan las 2 columnas con las caracteristicas 'es_versicolor' y 'es_virginica'
iris_df['es_versicolor'] = np.where( iris_df['especie'] == 'versicolor' , 1, 0)
iris_df['es_virginica'] = np.where( iris_df['especie'] == 'virginica' , 1, 0)

print('\nDataFrame Final:')
print(iris_df)


#---------------------------------------------------------------
# 8) - Listado de columnas
#--------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Listado de columnas')
print('#-----------------------------------------------------------------------------------')
print(set(iris_df.columns))


#---------------------------------------------------------------
# 9) - Relación lineal entre las medidas
#--------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Relación lineal entre las medidas de las flores')
print('#-----------------------------------------------------------------------------------')


def iris_corr_regr(var_ind, var_dep, especie):

    #Extracción de la especie requerida según los parametros de entrada.
    subDataFrame = iris_df[iris_df['especie'] == especie]
    data1 = subDataFrame[ var_ind ]    
    data2 = subDataFrame[ var_dep ]

    crecimiento, valor_inicial, correlación, p_valor, _ = stats.linregress(data1,data2)
    
    return crecimiento, valor_inicial, correlación, p_valor


print("crecimiento, valor_inicial, correlación, p_valor: {}".format(iris_corr_regr('longitud_sépalo', 'ancho_sépalo', 'virginica')))
