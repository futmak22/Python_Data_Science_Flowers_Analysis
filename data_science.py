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


#---------------------------------------------------------------
# 10) - Grafica relación lineal
#--------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Grafica relación lineal:')
print('#-----------------------------------------------------------------------------------')

# Defina los valores de entrada y ejecute la visualización (OPCIONAL)
var_ind = 'longitud_pétalo'
var_dep = 'longitud_sépalo'
especie = 'virginica'
print("Variables seleccionadas:")
print("Variable Independiente: " + var_ind)
print("Variable Dependiente: " + var_dep)
print("Especie: " + especie)

# Reutilizando la función iris_corr_regr().
m, b, r, p = iris_corr_regr(var_ind, var_dep, especie) 
mod_lin =  m * iris_df[var_ind] + b

# Gráfica de dispersión
graficar_relacion = input("Desea graficar la relación entre las variables?: (S/N) ")

if graficar_relacion == 'S' or graficar_relacion == 's':
    ax = iris_df.plot.scatter(var_ind, var_dep, figsize = (6,5)) 
    ax.get_figure().set_dpi(105);
    ax.plot(iris_df[var_ind], mod_lin, c='r', label='Regresión lineal'); 
    ax.legend();
    plt.show()


#---------------------------------------------------------------------------
#-- 11) - Prueba de Hipotesis - Nivel de confianza de la hipotesis
#---------------------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Prueba de Hipotesis - Nivel de confianza de la hipotesis:')
print('#-----------------------------------------------------------------------------------')
def iris_hipótesis_especies(variable, confianza):

    # Extracción de la caracteristica recibida por parametro para cada especie
    # Extracción para 'virginica'
    subDataFrame1 = iris_df[iris_df['especie'] == 'virginica']
    data1 = subDataFrame1[ variable ]
    
    # Extracción para 'versicolor'
    subDataFrame1 = iris_df[iris_df['especie'] == 'versicolor']
    data2 = subDataFrame1[ variable ]
    
    # Ejecución de la prueba de hipotesis (Objetivo: obtener p_valor que hace referencia al área bajo la curva de H1 que es la hipotesis de rechazo)
    # estadistico, p_valor = stats.ttest_rel( data1 , data2 )
    estadistico, p_valor = ztest( data1 , data2 )

    # El parametro de entrada 'confianza' es para afirmar que se presenta la Ho:(Hipotesis nula)
    # Debido a lo anterior es necesario hallar el valor restante que afirma que se debe negar Ho y afirmar H1.
    rechazo_hipotesis = 1 - confianza   # para 0.95% = 0.5, para 0.999% = 0.1 (Corresponde al porcentaje de rechazo)

    # Ho es la hipotesis nula
    # Ho: Los promedios de las 2 especies NO difieren significativamente. Entonces promedio1 = promedio2
    # H1: Los promedios de las 2 especies SI difieren significativamente. Entonces promedio1 < promedio2 ó promedio1 > promedio2 (La razon de dividir el Pvalor en 2)

    if p_valor > rechazo_hipotesis: # Si p_valor es mayor que el porcentaje de rechazo(rechazo_hipotesis) se pasa al lado de la aceptacion de Ho.
        resultado = 'ninguna'
    else: 
        #Si p_valor es menor o igual que el porcentaje de rechazo(rechazo_hipotesis) se queda en el lado del rechazo de Ho, es decir en H1.
        #H1: que los promedios de las 2 especies difieren significativamente
        promedio1 = data1.mean() #Calculo del promedio para 'virginica'
        promedio2 = data2.mean() #Calculo del promedio para 'versicolor'
        if promedio1 > promedio2:
            resultado = 'virginica'
        else:
            resultado = 'versicolor'

    return resultado

var = 'ancho_sépalo'
confianza = 0.95
print("Variable a evaluar: {}, nivel de confianza: {}, especie: {}".format( var , confianza , iris_hipótesis_especies(var, confianza)))

var = 'ancho_sépalo'
confianza = 0.999
print("Variable a evaluar: {}, nivel de confianza: {}, especie: {}".format( var , confianza , iris_hipótesis_especies(var, confianza)))


#--------------------------------------------------------------------------------------------------------
#-- 12) - Es posible distinguir entre las flores virginica y versicolor usando una regresion Logistica:
#--------------------------------------------------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Distinción entre Virgínica y Versicolor usando una REGRESION LOGISTICA:')
print('#-----------------------------------------------------------------------------------')
def iris_calcular_logit(especie, variable):
    
    # Creación de la formula del modelo:
    es_especie  = "es_" + especie
    myFormula = es_especie + " ~ " + variable
    
    # Creación del modelo:
    model  = sm.GLM.from_formula( myFormula , family = sm.families.Binomial(), data = iris_df)

    #Ejecución del entrenamiento del modelo
    result = model.fit()
    
    #Datos de Testing(Datos que serán clasificados durante el tiempo de ejecución)
    X = iris_df[ variable ] # longitud_sépalo, ancho_sépalo, longitud_pétalo ó ancho_pétalo

    #Ejecución de la clasificación de datos, sabiendo que el modelo ya fue entrenado.
    Valores_de_y_predic = result.predict(X)
    y = Valores_de_y_predic
        
    return y.values


# Diferentes pruebas modificando la selección de 'especie' y 'variable' a continuación:
especie = 'versicolor'
variable = 'longitud_pétalo'

y = iris_calcular_logit(especie, variable) # Resultado de ejecutar la regresión logística.

print("En este caso los resultados variarán en el rango [0, 1]:")
print("- El rango [0, 5) representará la una especie.")
print("- El rango [5, 1] representará la otra especie.")
print("Resultado de la clasificación por regresión logistica de los 1eros. 8 datos: {}".format(np.round(y[:8], 4)))


#--------------------------------------------------------------------------------------------------------
#-- 13) - Grafica de la regresión logística, generada anteriormente:
#--------------------------------------------------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Grafica de la REGRESION LOGISTICA:')
print('#-----------------------------------------------------------------------------------')
def graficar_regresion(y, especie, variable):
  ax = iris_df.plot.scatter(variable, f'es_{especie}', label = f"Especie real: {especie}", figsize = (6,4.5)); # Gráfica de dispersión con la variable objetivo.
  ax.get_figure().set_dpi(105)  
  ax.scatter(iris_df[variable], y, color = 'red', lw = 1, label = "Regresión logística"); # Gráfica de líneas con los puntos calculados con la regresión.
  ax.legend()
  plt.show()

# Graficar los resultados de la regresión logística (OPCIONAL)
reg_log = input("Desea graficar la regresión logística? (S/N): ")
if reg_log == 'S' or reg_log == 's':
    graficar_regresion(y, especie, variable)


#--------------------------------------------------------------------------------------------------------
#-- 14) - Evaluación de la predicción de la regresión logistica:
#--------------------------------------------------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Evaluación de la predicción de la regresión logistica:')
print('#-----------------------------------------------------------------------------------')
def iris_evaluar_logit(especie, variable):
  
  iris_df['modelo logit'] = iris_calcular_logit(especie, variable)
  iris_df['predicción']   = np.round(iris_df['modelo logit'], decimals=0)
  iris_df['acierto']      = np.where( iris_df[ "es_" + especie ] == iris_df[ 'predicción' ] , True, False)
   
  return iris_df

wrk_especie  = 'versicolor'
wrk_variable = 'longitud_pétalo'

print("Vista de los 1ros. 10 elementos resultado de la predicción:")
print(iris_evaluar_logit(wrk_especie, wrk_variable)[
      ['es_virginica', 'es_versicolor', 'predicción', 'acierto']
    ].head(10)
)
print('#-----------------------------------------------------------------------------------')

#--------------------------------------------------------------------------------------------------------
#-- 15) - Evaluación de la precisión:
#--------------------------------------------------------------------------------------------------------
print('\n')
print('#-----------------------------------------------------------------------------------')
print('#-Evaluación de la precisión:')
print('#-----------------------------------------------------------------------------------')
wrk_especie  = 'versicolor'
wrk_variable = 'longitud_pétalo'
iris_df_prediccion = iris_evaluar_logit(wrk_especie, wrk_variable)
accuracy = iris_df['acierto'].mean() # En la suma los valores booleanos True y False se interpretan como 1 y 0.
print("La regresión logística tuvo una presición de: {}".format(accuracy))
print('\n')