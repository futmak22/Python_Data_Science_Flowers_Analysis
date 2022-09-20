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

