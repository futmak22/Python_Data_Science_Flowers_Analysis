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

