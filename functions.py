'''
Librerias a utilizar
'''
import profile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import re
from plotly.offline import init_notebook_mode, iplot, plot
from matplotlib import cm
import joypy
from joypy import joyplot
import random
import wget
import pygame
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import cv2 as cv
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn import linear_model, metrics, model_selection
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix,r2_score, mean_absolute_error, explained_variance_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline 
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport
from skimage.io import imread
import os
import cv2
import numpy as np


## | JAVI |


def graficas (df,y):
    '''
    Función para representar varias graficas antes de realizar cualquier modelo
    df es un DataFrame.

    Args:
        df:Dataframe con las variables numéricas
        y: Variable target.

    Return:
        Subplot compuesto por:
        Gráfica 'pairplot'
        Gráfica 'heatmap'
    '''
    plt.figure(figsize=(20,20))
    sns.pairplot(df)
    plt.fig, axes = plt.subplots(2,1)
    sns.distplot(y, ax = axes[0])
    sns.heatmap(df.corr(), annot=True, ax = axes[1])
    axes[0].set_title("Distribucion")
    axes[1].set_title("Mapa Correlación");

def funcion_lineal_regression(X,y,test_size_1:float,random_state_1:int):
    '''
    Función para ingresar los datos de las variables previsoras (X) y la variable target (y), 
    los parámetros necesarios para realizar el train test split (random_state y test_size).

    Args:
        X:Dataframe con las variables predictoras
        y: Variable target
        test_size:float, porcentaje designado para test
        random_state: semilla para reproducir la función.

    Return:
        Devuelve las siguientes variables:
        X_train: Dataframe de las variables predictoras para el entrenamiento del modelo de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_train: Dataframe de las variables target para el entrenamiento del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        lin_reg: módelo entrenado
        lin_reg.intercept_: float, valor de ordenadas de la función de regresión lineal
        lin_reg.coef_: lista, coeficientes de la función de regresión lineal
        coeff_df: Dataframe con los coeficientes de la función de regresión lineal.
    '''
    
    lin_reg = LinearRegression()   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_1, random_state=random_state_1)
    lin_reg.fit(X_train, y_train)                           #   Entrenas/generas el modelo para determinar los coeficientes

    print("Estos son los datos del test y del target:\n-----")
    print("Total features shape:", X.shape)
    print("Train features shape:", X_train.shape)
    print("Train target shape:", y_train.shape)
    print("Test features shape:", X_test.shape)
    print("Test target shape:", y_test.shape)  

    print("Estos son los datos del valor de y en x=0 y de las pendientes de cada gradiente de las variables:\n-----")
    print(lin_reg.intercept_)
    print(lin_reg.coef_)
    coeff_df = pd.DataFrame(lin_reg.coef_,
                            X.columns,
                            columns=['Coefficient'])
    print("Estos son las pendientes de cada gradiente visto en un Dataframe:\n-----")
    print(coeff_df)

    return X_train, X_test, y_train, y_test,lin_reg, lin_reg.intercept_,lin_reg.coef_,coeff_df

def función_metricas_error (model,X_test,y_test,X_train,y_train):
    '''
    Función que a partir de la función entrenada te facilita las métricas más importantes en regresión lineal.
    
    Args:
        model: modelo entrenado de regresión lineal
        X_train: Dataframe de las variables predictoras para el entrenamiento del modelo de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_train: Dataframe de las variables target para el entrenamiento del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal.

    Return:
        Devuelve las siguientes variables:
        mae_pred: Mean Absolute Error de las prediciones 
        mape_pred: Mean absolute percentage error de las predicciones
        mse_pred: Mean Squared Error de las prediciones
        msqe_pred: Mean Squared Quadratic Error de las predicciones
        mae_train: Mean Absolute Error del entrenamiento
        mape_train: Mean absolute percentage error del entrenamiento
        mse_train: Mean Squared Error del entrenamiento
        msqe_train: Mean Squared Quadratic Error de las entrenamiento.
    '''

    predictions = model.predict(X_test)                   #   Determino los resultados que deberían de dar con los valores guardados para
    score=model.score(X_test, y_test)
    mae_pred= metrics.mean_absolute_error(y_test, predictions)
    mape_pred=metrics.mean_absolute_percentage_error(y_test, predictions)
    mse_pred=metrics.mean_squared_error(y_test, predictions)
    msqe_pred=np.sqrt(metrics.mean_squared_error(y_test, predictions))

    mae_train= metrics.mean_absolute_error(y_train, model.predict(X_train))
    mape_train=metrics.mean_absolute_percentage_error(y_train, model.predict(X_train))
    mse_train=metrics.mean_squared_error(y_train, model.predict(X_train))
    msqe_train=np.sqrt(metrics.mean_squared_error(y_train, model.predict(X_train)))

    print("El factor de correlacion de la regresión es: ",score)
    print("Errores de las predicciones:\n---")
    print('MAE:', mae_pred)
    print('MAPE:', mape_pred)
    print('MSE:', mse_pred)
    print('RMSE:', msqe_pred)
    print("\nErrores de los tests\n---")
    print('MAE:', mae_train)
    print('MAPE:',mape_train)
    print('MSE:', mse_train)
    print('RMSE:', msqe_train)

    print("Esta es la importancia de las variables:\n-----")
    features = pd.DataFrame(model.coef_, X_train.columns, columns=['coefficient'])
    print(features.head().sort_values('coefficient', ascending=False))

    return mae_pred,mape_pred,mse_pred,msqe_pred,mae_train,mape_train,mse_train,msqe_train


def funcion_ridge (model,X_test,y_test,X_train,y_train,alpha_1):
    '''
    Función para entrenar la función de ridge y el calculo del error regularizando o sin regularizar del MSE.

    Args:
        model: modelo entrenado de regresión lineal
        X_train: Dataframe de las variables predictoras para el entrenamiento del modelo de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_train: Dataframe de las variables target para el entrenamiento del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        alpha_1:int. Número de variable alpha para entrenar la función Ridge.

    Return:
        RidgeR: función Ridge entrenada.
    '''
    
    ridgeR = Ridge(alpha = alpha_1)
    ridgeR.fit(X_train, y_train)

    mse_pred=metrics.mean_squared_error(y_test, model.predict(X_test))
    mse_train=metrics.mean_squared_error(y_train, model.predict(X_train))

    mse_ridge_pred=metrics.mean_squared_error(y_test, ridgeR.predict(X_test))
    mse_ridge_train=metrics.mean_squared_error(y_train, ridgeR.predict(X_train))

    print("------")
    print("Train MSE sin regularización:", round(mse_train,2))
    print("Test MSE sin regularización:", round(mse_pred,2))
    print("------")
    print("Train MSE:", round(mse_ridge_train,2))
    print("Test MSE:", round(mse_ridge_pred,2))
   
    return ridgeR


def funcion_lasso(model,X_test,y_test,X_train,y_train,alpha_1:int):

    '''
    Función para entrenar la función de Lasso y el calculo del error regularizando o sin regularizar del MSE.

    Args:
        model: modelo entrenado de regresión lineal
        X_train: Dataframe de las variables predictoras para el entrenamiento del modelo de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_train: Dataframe de las variables target para el entrenamiento del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        alpha_1:int. Número de variable alpha para entrenar la función Lasso.

    Return:
        LassoR: función Ridge entrenada.
    '''

    lassoR = Lasso(alpha=alpha_1)
    lassoR.fit(X_train, y_train)

    mse_pred=metrics.mean_squared_error(y_test, model.predict(X_test))
    mse_train=metrics.mean_squared_error(y_train, model.predict(X_train))

    mse_lasso_pred=metrics.mean_squared_error(y_test, lassoR.predict(X_test))
    mse_lasso_train=metrics.mean_squared_error(y_train, lassoR.predict(X_train))

    print("------")
    print("Train MSE sin regularización:", round(mse_train,2))
    print("Test MSE sin regularización:", round(mse_pred,2))
    print("------")
    print("Train MSE:", round(mse_lasso_train,2))
    print("Test MSE:", round(mse_lasso_pred,2))

    return lassoR

def correccion_ridge_a_aplicar(model, X_test, y_test, ridgeR, log_ini:int,log_fin:int,n_alphas:int):
    '''
    Función que evalua la regularización de Ridge para un modelo de regresión lineal entrenado y
    que a partir de los valores logarítmicos y alpha muestra una gráfica donde se puede localizar 
    el punto más bajo de los errores y así determinar cuál es el valor de alpha más adecuado.

    Args:
        model: modelo entrenado de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        ridgeR: función de Ridge entrenada
        log_ini:int, valor inicial logarítmica desde donde empezar a evaluar la función Ridge para conseguir el menor alpha
        log_fin:int, valor final logarítmica desde donde empezar a evaluar la función Ridge para conseguir el menor alpha
        n_alphas:int. Número de variable alpha a usar para optimizar la función Ridge.

    Return:
        Grafica: muestra los valores de alpha en abscisas en el rango indicado y los valores de Mean Square Error de la función.

    
    OJO!!! esta función está por revisar'''
    predictions = model.predict(X_test)                   #   Determino los resultados que deberían de dar con los valores guardados para

    alphas = np.logspace(log_ini, log_fin, n_alphas) 
    baseline_error = metrics.mean_squared_error(y_test, predictions)
    coef_ridgeR = []
    err_ridge = []
    baseline = []

    for a in alphas:
        ridgeR.set_params(alpha=a)
        coef_ridgeR.append(ridgeR.coef_)
        y_pred = ridgeR.predict(X_test)
        lasso_error = metrics.mean_squared_error(y_pred, y_test)    
        err_ridge.append(lasso_error)
        baseline.append(baseline_error)
    print(min(err_ridge))
    
    plt.figure(figsize=(20,12))
    ax = plt.gca()
    ax.plot(alphas, err_ridge, linewidth=5, color='red', label="Ridge regression")
    ax.plot(alphas, baseline, linewidth=4,linestyle='--', color='blue', label='Linear regression')
    ax.set_xscale('log')
    plt.xlabel('$\lambda$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('error', fontsize=30)
    ax.legend(fontsize=30)
    plt.title(r'Regression error ($\lambda$)', fontsize=30)
    plt.show();

def correccion_Lasso_a_aplicar(model, X_test, y_test, LassoR, log_ini:int,log_fin:int,n_alphas:int):
    '''
    Función que evalua la regularización de Lasso para un modelo de regresión lineal entrenado y
    que a partir de los valores logarítmicos y alpha muestra una gráfica donde se puede localizar 
    el punto más bajo de los errores y así determinar cuál es el valor de alpha más adecuado.

    Args:
        model: modelo entrenado de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        LassoR: función de Lasso entrenada
        log_ini:int, valor inicial logarítmica desde donde empezar a evaluar la función Ridge para conseguir el menor alpha
        log_fin:int, valor final logarítmica desde donde empezar a evaluar la función Ridge para conseguir el menor alpha
        n_alphas:int. Número de variable alpha a usar para optimizar la función Ridge.

    Return:
        Grafica: muestra los valores de alpha en abscisas en el rango indicado y los valores de Mean Square Error de la función.

    OJO!!! esta función está por revisar
    '''

    predictions = model.predict(X_test)
    alphas = np.logspace(log_ini, log_fin, n_alphas) 
    baseline_error = metrics.mean_squared_error(y_test, predictions)
    coef_lassoR = []
    err_lasso = []
    baseline = []

    for a in alphas:
        LassoR.set_params(alpha=a)
        coef_lassoR.append(LassoR.coef_)
        y_pred = lassoR.predict(X_test)
        lasso_error = metrics.mean_squared_error(y_pred, y_test)    
        err_lasso.append(lasso_error)
        baseline.append(baseline_error)


    print(min(err_lasso))
    plt.figure(figsize=(20,10))
    ax = plt.gca()
    ax.plot(alphas, err_lasso, linewidth=5, color='red', label="Lasso")
    ax.plot(alphas, baseline, linewidth=4,linestyle='--', color='blue', label='Linear regression')
    ax.set_xscale('log')
    plt.xlabel('$\lambda$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylabel('error', fontsize=30)
    ax.legend(fontsize=30)
    plt.title(r'Regression error ($\lambda$)', fontsize=30)
    plt.show();

def error_modelo(model, X_test, y_test):
    '''
    Función que a partir de un modelo entrenado con las variables X_test e y_test, muestra las
    métricas más relevantes de un módelo clasificatorio.

    Args:
        model: modelo entrenado de regresión clasificatorio
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión clasificatorio
        y_test: Dataframe de las variables target para el testeo del modelo de regresión clasificatorio.

    Return:
        df_error: Dataframe donde aparecen los datos de 'Accuracy','f-1 score','Recall','Precision'.
        Muestra también el cálculo de la curva ROC
        Grafica de la matriz de confunsión. 

    '''

    y_pred = model.predict(X_test)
    f1_model=f1_score(y_test, y_pred,average='macro')
    acc_model=accuracy_score(y_test, y_pred)
    precision_model=precision_score(y_test, y_pred,average='macro')
    recall_model=recall_score(y_test, y_pred,average='macro')
    roc_auc_score=roc_auc_score(y_test, model.predict_proba(X_test),multi_class='ovr')
    conf_model=confusion_matrix(y_test, y_pred, normalize='true')
    model_error = {'accuracy': acc_model, 'f-1': f1_model, 'recall': recall_model , 'precision': precision_model}
    df_error=pd.DataFrame.from_dict(model_error,orient='index')

    print('Accuracy', acc_model)
    print('F1', f1_model)
    print('Precision', precision_model)
    print('Recall', recall_model)
    print('-'*30)
    print('ROC', roc_auc_score)

    plt.figure(figsize=(10,10))
    sns.heatmap(conf_model, annot=True)
    return df_error


## | LUIS | 20_09_14_28

def time_now():
    """
    Función que devuelve la fecha y hora actual

    Args: No tiene parámetros.

    Return(tuple): Tupla de strings con el día de la semana, día del mes, mes, año, hora, minuto y segundo actual.
    """

    dt = datetime.now()

    dia = str(dt.day).zfill(2)
    mes = str(dt.month).zfill(2)
    anyo = str(dt.year).zfill(2)
    hora = str(dt.hour).zfill(2)
    minuto = str(dt.minute).zfill(2)
    segundo = str(dt.second).zfill(2)

    hoy = datetime.today()
    diaSemana = hoy.strftime("%A")

    return diaSemana, dia, mes, anyo, hora, minuto, segundo

def feature_visual(url):
    '''Función que permite importar el archivo csv y devolver un analisis de cada columna del dataframe.
(Comparativa por columnas, mapa de calor, mapa de correlaciones.)'''    
    df=pd.read_csv(url)
    profile=ProfileReport(df, title="Pandas Profiling Report")
    return print(profile)


    

## | SARA | 20_09_14_28

def grafico_goscatter(df, columna_eje_x, columna_eje_y, color, texto_labels):
    '''
    Función para crear un gráfico PLOTLY scatter de tipo lineal
    a partir de una columna de un dataframe,
    definiendo las columnas en el eje x e y, 
    el color de la línea y la etiqueta de la misma.

    Args:
        df: dataframe
        columna_eje_x: columna del dataframe que aparecerá en el eje X
        columna_eje_y: columna del dataframe que aparecerá en el eje y
        color: color de la línea
        texto_labels: texto que aparece cuando pasamos por encima el cursos
    
    Returns:
        Devuelve la gráfica.
    '''

    trace = go.Scatter(
                x = df[columna_eje_x],
                y = df[columna_eje_y],
                mode= 'lines',
                marker = dict(color = color),
                texttemplate="simple_white",
                text = df[texto_labels])
    fig = go.Figure(data = trace)
    iplot(fig)


def sustituye_texto(df, columna, condicion, reemplazo):
    '''
    Función para sustituir texto por columnas,
    donde se cumpla una condición mediante == (máscara).

    Args:
        df: dataframe
        columna: columna del dataframe
        condicion: lo que queremos sustituir
        reemplazo: lo que queremos que aparezca

    Return:
        Dataframe modificado.

    Ejemplos:
    * Con strings
        df[df['personaje']==Monica]=df[df['personaje']==Monica].apply(lambda x: x.replace('Monica', 'Monica Geller))

    * Con int o float
        df[df['money']==80]=df[df['money']==80].apply(lambda x: x.replace(80, 100))
    '''

    df[df[columna]==condicion]=df[df[columna]==condicion].apply(lambda x: x.replace(condicion, reemplazo))
    return df


def extraer_con_regex(df, columna, clave_regex):
    '''
    Función para quedarnos con la parte del texto o del dato que queramos
    mediante regex. 
    La columna existente se reemplaza por el resultado después de aplicar
    regex.
    *Formato para la clave_regex = '(regex)'

    Args:
        df: dataframe
        columna: columna del dataframe
        clave_regex: la clave regex que seleccione lo que nos queremos quedar.
             *Formato para la clave_regex = '(regex)'
 
    Return:
        Dataframe modificado.

    Ejemplos:
        df['personaje'] = df['personaje'].str.extract(r'(^\w+)')
    '''
    
    df[columna] = df[columna].str.extract(clave_regex)
    return df


def eliminar_entre_parentesis_en_df(df, columna):
    '''
    Función para eliminar texto entre paréntesis (acotaciones).
    Se aplica sobre toda la columna del dataframe.
    
    Args:
        df: dataframe
        columna: columna del dataframe

    Return:
        Dataframe modificado.
    '''

    for i in range(len(df[columna])):
        if '(' in df[columna][i]:
            df[columna][i] ="".join(re.split("\(|\)|\[|\]", df[columna][i]))
    return df


def where_contains(df, columna, palabra_clave):
    '''
    Función para crear columnas nuevas en un dataframe,
    a partir de si en otra columna está o no la palabra_clave.
    En caso de que esté la palabra, la columna nueva generará un 1.
    En caso de que no esté, generará un 0.

    Solo es válida para strings.
    
    Args:
        df: dataframe
        columna: columna del dataframe
        palabra_clave: string

    Return:
        Dataframe modificado.

    Ejemplo:
        df['details']= np.where((df['details'].str.contains('hidromasaje')),1,0)
    '''

    df[columna]= np.where((df[columna].str.contains(palabra_clave)),1,0)
    return df


def drop_con_condicion(df, columna, condicion):
    '''
    Funcion para eliminar los registros de una columna que cumplen
    una condición de tipo == condicion.
        
    Args:
        df: dataframe
        columna: columna del dataframe
        condicion: lo que tienen que cumplir los registros que queremos eliminar

    Return:
        Dataframe modificado.
    '''

    df.drop(df[df[columna]==condicion].index, inplace=True)
    return df


## | IRENE | 20_09_15_00

def data_report(df):

    '''
    Genera DF cuyas columnas son las del df, y filas q indican el tipo de dato, porcentaje de missings, número de valores únicos 
    y porcentaje de cardinalidad.

    Arg:
        df: dataframe

    Return:
        Dataframe de valor informativo
    '''

    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T


def number_of_outliers(df):
    
    '''
    Devuelve la suma de outliers en cada columna

    Arg:
        df: dataframe 

    Return:
        Imprime por pantalla la suma de outliers para cada columna
    '''
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()


def radical_dropping(df):

    '''
    Elimina todos los missings y duplicados

    Arg:
        df: dataframe

    Return:
        Dataframe modificado
    '''

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)


        
## | MARIO |

    
def read_data_bw(path, im_size, class_names_label):

    '''Lectura y etiquetado de imágenes en blanco y negro.

    Args:
        path(str): ruta donde estarán el resto de carpetas.

        im_size(tuple): tamaño al que queremos pasar todas las imagenes.

        class_names_label(dict): nombre de las variables a etiquetar.
      
    Return:
        X: el array de los datos de las imágenes.

        Y: array con los label correspondientes a las imágenes.
    '''
    X = []
    Y = []
    
    for folder in os.listdir(path):
        print('Comenzamos a leer ficheros de la carpeta', folder)
        label = class_names_label[folder]
        folder_path = os.path.join(path,folder)
        ##### CODE #####
        # Iterar sobre todo lo que haya en path
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)

            # Leer la imagen en blanco y negro
            image = imread(image_path)
            
            # Resize de las imagenes
            smallimage = cv2.resize(image, im_size)
            
            # Guardo en X e Y
            X.append(smallimage)
            Y.append(label)
        print('Terminamos de leer ficheros de la carpeta', folder)
        

    return np.array(X), np.array(Y)
    # Ejemplo de class_names_label: tipo diccionario
    # class_names_label {'angry': 0,'disgust': 1,'fear': 2,'happy': 3,'neutral': 4,'sad': 5,'surprise': 6}



def read_data_color(path, im_size, class_names_label):

    '''Lectura y etiquetado de imágenes a color.

    Args:
        path(str): ruta donde estarán el resto de carpetas.

        im_size(tuple): tamaño al que queremos pasar todas las imagenes.

        class_names_label(dict): nombre de las variables a etiquetar.
      
    Return:
        X: el array de los datos de las imágenes.

        Y: array con los label correspondientes a las imágenes.
    '''

    X = []
    Y = []
    
    for folder in os.listdir(path):
        print('Comenzamos a leer ficheros de la carpeta', folder)
        label = class_names_label[folder]
        folder_path = os.path.join(path,folder)
        ##### CODE #####
        # Iterar sobre todo lo que haya en path
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            # Leer la imagen a color y aplicarle el resize
            image = imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            smallimage = cv2.resize(image, im_size)
            
            # Guardo en X
            X.append(smallimage)
            Y.append(label)
        print('Terminamos de leer ficheros de la carpeta', folder)
        

    return np.array(X), np.array(Y)


def read_data(path):
    '''Lectura de imágenes de una carpeta.

    Args:
        path(str): ruta donde están las imágenes.
      
    Return:
        X: el array de los datos de las imágenes.

    '''
    
    X = []
    for file in os.listdir(path):
        image = imread(path + '/' + file)
        smallimage = cv2.resize(image, (224,224))
        print(path + '/' + file)

        X.append(smallimage)
    return np.array(X)

## | Xin |

def gen_diagrama_caja(df):

    """Funcion que genera n diagramas de caja segun el numero de columnas numericas que contiene el datafreme.

    Args: 
        df : datafreme

    Return: 
        n diagrama de caja 

    """
    num_cols = df.select_dtypes(exclude='object').columns
    for col in num_cols:
        fig = plt.figure(figsize= (5,5))
        sns.boxplot(x=df[col],)
        fig.tight_layout()  
        plt.show()



def sustituir_outliers(df, col):
    """Funcion que detecta los outliers del datafreme, y lo sustituye por la media. 

    Args: 
        df : datafreme
        col : la columna que contiene outliers

    Return: 
        datafreme sustituido

    """

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    intraquartile_range = q3 - q1
    fence_low  = q1 - 1.5 * intraquartile_range
    fence_high = q3 + 1.5 * intraquartile_range
    df[col]=np.where(df[col] > fence_high,df[col].mean(),df[col])
    df[col]=np.where(df[col] < fence_low,df[col].mean(),df[col])

    return df


def muestra_nan(df):

    """Funcion que muestra los missing values de cada columna de detafreme y el porcentaje de missing values.

    Args: 
        df : datafreme

    Return: 
        detafreme : datafreme nuevo donde muestra el porcentaje de missing values. 

    """
    suma_nan = df.isnull().sum().sort_values(ascending = False)
    percentaje_nan = (df.isnull().sum() / df.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([suma_nan, percentaje_nan], axis=1, keys = ['suma_nan', 'percentaje_nan'])


#|| LAURA ||

def pieplot_one_column(dataframe, column, title, background_colour, colour_map=None):
    '''
    Función para representar un pie plot, mostrando el value.counts de una columna
    Permite personalizar paleta de colores, color de fondo y título.
    Args:
        dataframe: Dataframe a utilizar (dataframe)
        column: Nombre de la columna para hacer el value counts (str)
        title: Título de la figura (str)
        background_colour: Color de fondo en formate HEX o palabra (str)
        colour_map: Mapa de color de matplotlib en formato: cm.seismic, las paletas se 
                    pueden encontrar en https://matplotlib.org/stable/tutorials/colors/colormaps.html
    Return:
        Gráfica pieplot
    '''
    fig, ax = plt.subplots(facecolor=background_colour, figsize=(13, 8))
    data = dataframe[column].value_counts()
    data.plot(kind='pie', autopct='%.0f%%', wedgeprops={"edgecolor":"white"},colormap=colour_map)
    plt.legend(loc = 2, bbox_to_anchor = (1,1), prop={'size': 15}, facecolor=background_colour, edgecolor='white', )
    plt.title(title, pad=30, fontsize = 15)
    plt.show();

def joyplot_one_column(dataframe, classifier_column, numeric_column, title, line_colour='white', colour_map=None, 
                       figsize=(7,4), x_limit=None):
    '''
    Función para representar un joyplot, mostrando los valores de una columna, agrupados por otra
    Permite personalizar paleta de colores, título y color de linea.
    Args:
        dataframe: Dataframe a utilizar (dataframe)
        classifier_column: Columna por la que se quiere hacer un groupby (str)
        numeric_column: Columna de la que se analizarán los datos
        title: Título de la figura (str)
        line_colour: Color de línea de contorno, en formato HEX o palabra (str). Por defecto es 'white'
        colour_map: Mapa de color de matplotlib en formato: cm.seismic, las paletas se 
                    pueden encontrar en https://matplotlib.org/stable/tutorials/colors/colormaps.html
                    Por defecto es la paleta automática de matplotlib
        figsize: Tamaño de la figura, en formato tupla. Por defecto es (7,4)
        x_limit: Límites para el eje x, formato lista. Por defecto es None.
    Return:
        Gráfica joyplot
    '''
    fig, axes = joyplot(dataframe, by = classifier_column, column = numeric_column, colormap=colour_map, fade = True, 
    figsize = figsize, title = title, linewidth=0.4, linecolor=line_colour)
    for a in axes[:-1]:
        a.set_xlim(x_limit)  
    plt.show()


def acotar_valores_clase(dataframe, columnaclases, clase, columna_a_filtrar, max_val, min_val):
    '''
    Función para acotar el rango de valores de una determinada columna, haciendo una máscara por cada 
    clase, o valor de otra columna. Las filas por encima y por debajo de los valores dados se eliminarán.
    Args:
        dataframe: Dataframe a utilizar (dataframe)
        columna_clases: Columna categórica que contiene las clases con las que queremos crear máscaras para filtrar (str)
        clase: Elemento dentro de la columna de clases, con el que se quiere realizar la máscara (str)
        columna_a_filtrar: Columna numérica con la cual se quieren acotar los valores númericos (str)
        max_val: Valor máximo, los valores de la columna a filtrar que estén por encima se eliminarán (num)
        min_val: Valor mínimo, los valores de la columna a filtrar que estén por debajo se eliminarán (num)
    Return:
        Elimina los valores por encima y por debajo de cierto valor
    '''
    clase = dataframe.loc[dataframe[columnaclases] == clase, columna_a_filtrar]
    above_threshold = clase[clase > max_val].index.tolist()
    below_threshold = clase[clase < min_val].index.tolist()
    indexNames = above_threshold + below_threshold
    dataframe.drop(indexNames, inplace=True)


def wrap_perspective_cv2(src, strength):
    '''
    Función que utiliza OpenCV para aplicar una distrosión a una imagen (wrap perspective)
    Args:
        src: Array de píxels de la imagen a transformar 
        strength: Nivel de distorsión, a más alto, mayor distorsión (int)
    Return:
        Devuelve la imagen distorsionada
    '''
    image = src
    pts_base = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    pts1=np.random.rand(4, 2)*random.uniform(-strength,strength)+pts_base
    pts1=pts1.astype(np.float32)
    M = cv.getPerspectiveTransform(pts1, pts_base)
    trans_img = cv.warpPerspective(image, M, (src.shape[1], src.shape[0]))
    trans_img = cv.warpPerspective(image, M, (src.shape[1], src.shape[0]))

    return trans_img


def sql_rules():
    '''
    Función que hace sonar el estribillo de "No te olvides de poner el where en el delete from.
    Primero descarga el audio en el directorio actual, y después la hace sonar a través de pygame.
    Args: Sin argumentos.
    Return:
        Suena el estribil
    '''
    # get current directory and create path
    current_dir = os.getcwd()
    if '/Users' in current_dir:
        path = current_dir + '/noteolvidesdeponerelwhere.mp3'
    else:
        # AQUÍ HAY QUE CAMBIAR LA / EN EL NOMBRE DEL ARCHIVO PARA QUE SEA PARA WINDOWS, NO SÉ SI HAY QUE HACER OTRO PARA LINUX
        path = current_dir + '/noteolvidesdeponerelwhere.mp3'

    # download the wav file in your current directory
    url = 'https://drive.google.com/uc?export=download&id=1_2xEhK3rBiG8XaNJTymTy7TpDXGEo7Id'
    wget.download(url, path)

    # initiate pygame and play sound
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.Sound.play(pygame.mixer.Sound(path))


## | QINGHUA |


def feature_important(model,X):
    '''
    funcion que saca feature important del modelo  y su grafico

    Args:
        model: el modelo
        X: datafeme de los features

    Returns:
        datafreme de feature impottant
        grafico de feature impottant
    '''
    df=pd.DataFrame(model.feature_importances_,
                X.columns,
                columns = ["Feature imp"]).sort_values("Feature imp", ascending=False)
    grafico=df.sort_values("Feature imp").plot.barh(y='Feature imp')
    return (df,grafico)


def subplots(df,X,y1,y2):
    '''
    función que hace un subplot de una variable, distribuida según los datos de otras dos variables
    
    Args:
        x:columa elegido para x
        y1:columa elegido para y de scatterplot
        y2:columa elegido para y de  lineplot

    Returns:
        grafico subplot 

    '''
    f,(axi1,axi2)=plt.subplots(2,1 ,figsize=(10,10))
    sns.scatterplot(x=X,y=y1,data=df,ax=axi1)
    sns.lineplot(x=X,y=y2,data=df,ax=axi2)


def graf_displot(df):
    '''
    funcion que genera n graficos de distribucion segun las columnas numericas que tiene

    Args:
        df: datafreme
    
    Returns:
        n graficos de distribucion

    '''
    numCols = df.select_dtypes(exclude='object').columns
    for col in numCols:
        plt.figure(figsize=(10,10))
        sns.displot(x=col,data=df, palette=["#ff006e", "#83c5be", "#3a0ca3"])
        plt.show()

## Christian

def train_sampler (X_train, y_train,randomstate,scalertype,sampletype):
    """ Función para realizar over o undersampling o randomsampling para datos no balanceados.\n
        Se realiza después del train test split.

        Args:
            X_train (array)  : valores de X_train
            y_train (array)  : valores de y_train
            randomstate (int) : valor del randomstate
            scalertype (str) : nombre del scaler:  minmax , standard
            sampletype (str) : nombre del sampler : over, under , random

        Returns:
            X_train_res (array) : nuevo array del X_train scaled y sampled
            y_train_res (array) : nuevo array del y_train scaled y sampled
    """
    
    if scalertype == "minmax":
        scaler = MinMaxScaler()
    elif scalertype == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train_scal = scaler.fit_transform(X_train)  # Valor mínimo 10 --> 0, Valor máximo 50 --> 1
    print ("data scaled with scaler:", scaler)

    over = RandomOverSampler (random_state = randomstate)
    under = RandomUnderSampler(random_state = randomstate)
    

    if sampletype == "over":
        steps = [('o',over)]  
    elif sampletype == "under":
        steps = [('u',under)]
   

    pipeline1 = Pipeline(steps=steps)
    X_train_res, y_train_res = pipeline1.fit_resample(X_train_scal, y_train)

    shape1 = X_train_res.shape
    shape2 =y_train_res.shape
    print(f'After scaling and {sampletype} -sampling, the shape of train_X: {shape1}')
    print(f'After scaling and {sampletype} -sampling, the shape of train_y: {shape2}')
    print ("applied Methods: ",steps)


    return  X_train_res,  y_train_res



def string_replacer (df,col,replacestring,newvalue):
    """ Reemplaza un string deseado por otro string deseado en toda la columna.

        Args:
            df (DataFrame) :   Dataframe en que se debe aplicar
            col (str) :        Nombre de la columna
            replacestring (str) :  El string que debe ser reemplazado
            newvalue (str) : El nuevo valor 

        Returns:
            df[co] (array):  Array de la columna actualizado
     """
    df[col] = df[col].apply(lambda x : x.replace(replacestring,newvalue) )
    return df[col]






def basic_encoding (df):
    """
    Realiza el encoding de variables categorícas en númericas de manera simple,  \n 
    sin agregar nuevas columnas.

    Args:
        df (DataFrame) : DataFrame actual

    Returns:
        df (DataFrame) : Devuelve nuevo DataFrame

    
    """
    le = LabelEncoder()
    for i in df.columns:
            if df[i].dtype == 'object':
                    enc_name = i+"_encoded"
                    df[enc_name] = le.fit_transform(df[i])

    return df


def clean_emoji(text):
    ''' Funcion para limpiar los emojis que aparecen dentro de un texto.
    
        Args:
            text (str): texto sobre el que se pretende realizar la función.
        
        Returns:
            emoj_text (text): el texto que se introduce en el argumento pero 
            sin ningún emoji.
    
    '''
    emoji_text = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_text.sub(r'', text)


def nine_Regressor_Models( X_train, y_train, X_test, y_test):
    '''Función para aplicar los modelos KNeighborsRegressor, GradientBoostingRegressor, ExtraTreesRegressor, .\n
        RandomForestRegressor, RandomForestRegressor, DecisionTreeRegressor y LinearRegression.
       
        Args:
            X_train (array o dataFrame): valores de X_train
            y_train (array o dataFrame): valores de y_train
            X_test (array o dataFrame): valores de X_train
            y_test (array o dataFrame): valores de y_train
       
        Returns:
            la función imprime:
            Modelo
            Training time
            Explained variance
            Mean absolute error
            R2 score
    '''
    
    lista_modelo = []
    lista_precision =[]
    lista_mae=[]
    lista_varianza=[]
   
    
    regressors = [
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
        ExtraTreesRegressor(),
        RandomForestRegressor(),
        DecisionTreeRegressor(),
        LinearRegression()
    ]

    head = 10
    for model in regressors[:head]:
        start = time()
        model.fit(X_train, y_train)
        train_time = time() - start
        start = time()
        y_pred = model.predict(X_test)
        predict_time = time()-start    
        lista_modelo.append(model)
        lista_precision.append(r2_score(y_test, y_pred))
        lista_mae.append(mean_absolute_error(y_test, y_pred))
        lista_varianza.append(explained_variance_score)
        
        print(model)
        print("\tTraining time: %0.3fs" % train_time)
        print("\tPrediction time: %0.3fs" % predict_time)
        print("\tExplained variance:", explained_variance_score(y_test, y_pred))
        print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
        print("\tR2 score:", r2_score(y_test, y_pred))
        print()
        
        
def drop_outliers(df, field_name):
    ''' Esta función borra los outliers de la columna (field_name) del dataSet(df)

        Args:
            df (dataFrame): dataFrame original
            field_name (pandas.core.series): columna original
        
        Returns:
            df (dataFrame): nuevo dataFrame sin outliers en field_name
    '''
    
    iqr = 1.5 * (np.percentile(df[field_name], 75) - np.percentile(df[field_name], 25))
    
    try:
        df.drop(df[df[field_name] > (iqr + np.percentile(df[field_name], 75))].index, inplace=True)
    except:
        pass
    try:
        df.drop(df[df[field_name] < (np.percentile(df[field_name], 25) - iqr)].index, inplace=True)
    except:
        pass

    return df

## | Antonio |
def PruebaModelos(xtrain, ytrain, xtest, ytest, ModelosRegresion = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), DecisionTreeRegressor(), RandomForestRegressor(), ExtraTreesRegressor(), KNeighborsRegressor(), SVR()], 
ModelosClasificacion = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), KNeighborsClassifier(), SVC()], 
agregar = [], quitar = [], metricas = [], tipo = "regresion"):
    """Función para probar un conjunto de modelos de regresión y clasificación con los parámetros por defecto devolviendo las metricas de precisión de cada modelo.
        Esto es útil para hacerse una primera idea de hacia cuál modelo poder enfocarse
    
    Args:
        xtrain(array): Variables predictoras usadas para entrenar el modelo.

        ytrain(array): Variable a predecir usada para entrenar el modelo.

        xtest(array): Variables predictoras con las que predecir y comprobar la eficacia del modelo.

        ytest(array): Variable predictora con la que comporbar la eficacia del modelo.


        ModelosRegresión(list): por defecto, LinearRegression, Ridge, Lasso, ElasticNet, DecisionTreeRegressor, RandomForestRegressor, ExtraTreesRegressor, KNeighborsRegressory SVR.

        Se puede pasar otra lista entera si se desea.

        ModelosClasificacion(list):por defecto, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, KNeighborsClassifier y SVC

        Se puede pasar otra lista entera si se desea.

        agregar(list): Agrega un nuevo modelo para entrenar a lo que vienen por defecto.

        quitar(list): Quita un modelo para entrenar de los que vienen por defecto.

        metricas(list): Añade una metrica nueva para ver la eficacia del modelo(Para un tipo de modelo de regresión se puede añadir MAPE y para uno de clasificación se puede añadir Recall)

        tipo(str): Elige entre si quieres entrenar modelos de regresión o de clasificación. Las opciones son "regresión" o "clasificacion".
    
    Return:None
    """
    medidas = []
    resultado = ""
    if tipo == "regresion":
        for i in agregar:
            ModelosRegresion.append(i)
        for i in quitar:
            ModelosRegresion.remove(i)
        for modelo in ModelosRegresion:
            if modelo != SVR():
                modelo.fit(xtrain, ytrain)
            else:
                stdr = StandardScaler.fit_transform(xtrain)
                modelo.fit(stdr, ytrain)
            if metricas == []:
                medidas.append(str("MAE" + " " + str(modelo)[:-2] + ":" + " " + str(metrics.mean_absolute_error(ytest, modelo.predict(xtest)))))
                medidas.append(str("MSE" + " " + str(modelo)[:-2] + ":" + " " + str(metrics.mean_squared_error(ytest, modelo.predict(xtest)))))
            elif metrics.mean_absolute_percentage_error() in metricas:
                medidas.append(str("MAPE" + " " + str(modelo)[:-2] + ":" + " " + str(metrics.mean_absolute_percentage_error(ytest, modelo.predict(xtest)))))
            else:
                print("Metrica inválida")
                break
    elif tipo == "clasificacion":
        for i in agregar:
            ModelosClasificacion.append(i)
        for i in quitar:
            ModelosRegresion.remove(i)
        for modelo in ModelosClasificacion:
            if modelo != SVC():
                modelo.fit(xtrain, ytrain)
            else:
                stdr = StandardScaler.fit_transform(xtrain)
                modelo.fit(stdr, ytrain)
            if metricas == []:
                medidas.append(str("Accuracy" + " " + str(modelo)[:-2] + ":" + " " + str(metrics.accuracy_score(ytest, modelo.predict(xtest)))))
                medidas.append(str("Precission" + " " + str(modelo)[:-2] + ":" + " " + str(metrics.precision_score(ytest, modelo.predict(xtest)))))
            elif metrics.recall_score() in metricas:
                medidas.append(str("Recall" + " " + str(modelo)[:-2], + ":" + " " + str(metrics.recall_score(ytest, modelo.predict(xtest)))))
            else:
                print("Metrica inválida")
                break
    else:
        print("Tipo de modelo inválido")
    # print(medidas[0])
    for m in medidas:
        resultado = resultado + m + "\n"
    print(resultado)

def MinMaxCorr(data, min, max = None):
    """Función para buscar las columnas que tienen una correlación dentro de un rango. Su uso común es buscar las columnas que tienen mayor correlación entre ellas la cual sea
        diferente de 1.
    
    Args:
        data(DataFrame): DataFrame del cual obtener la correlación.

        min(int): Número mínimo de correlación que buscar.

        max(int): Número máximo de correlación que buscas. Por defecto tiene None y buscará todos los valores por encima del parámetro min que no sea 1.bit_length
    Return: Matriz de correlación con las columnas que tienen más correlación entre ellas
    """
    if max == None:
        resultado = data.corr()[(data.corr() > min) & (data.corr() != 1)].dropna(axis = 1, how = "all").dropna(axis = 0, how = "all")
    else:
        resultado = data.corr()[(data.corr() > min) & (data.corr() < max)].dropna(axis = 1, how = "all").dropna(axis = 0, how = "all")
    return resultado

def root_mean_squared_error(y_true, y_pred):
    """Función para añadir la métrica de RMSE que no viene por defecto en Sklean.
    
    Args:
        y_true(array): Variable objetivo con los valores correctos.

        y_pred(array): Variable objetivo con valores predichos por el modelo deseado.
    
    Return:int
    """
    return np.square(metrics.mean_squared_error(y_true, y_pred))

def DfTransType(data, type1 = "object", type2 = "float64"):
    """Función para transoformar todas las columnas un DataFrame con un tipo de dato concreto a otro.
    
    Args:
        data(DataFrame): DataFrame al cual transformar sus columnas.

        type1(str): Tipo de dato a cambiar. Por defecto es "object".

        type2(str): Tipo de dato al cual cambiar. Por defecto es "float64".
    
    Return:None
    """
    for i in data.dtypes[data.dtypes == type1].index:
        data[i] = data[i].astype(type2)


## Tarik

def sustituye_nan_moda(data):
    '''
    Funcion que rellena e iguala los valores de las columnas con la moda,
    para la correcta visualización y estudio del dataset.
    
    Args:
        data = dataset que contiene los datos con objeto de estudio.
    
    Return: dataframe listo para su estudio y visualización.
    '''
    iguala = [column for column in data.columns if data[column].isna().sum() > 0]

    for column in iguala:
        data[column] = data[column].fillna(data[column].value_counts().index[0])



def train_regression(model, xtrain, ytrain, xtest, ytest):
    '''
    Funcion que entrena modelo de regresión y devuelve las metricas.
    Args:
        model(model) = modelo que vamos a entrenar.
    
        xtrain, ytrain, xtest, ytest = los valores que vamos a entrenar, para entrenar el modelo.
    
    Return: Devuelve las predicciones del model
    
    '''
    model.fit(xtrain,ytrain)
    print("intercepto:", model.intercept_)
    print("coeficientes:", model.coef_)
    mp = model.predict(xtest)
    print('-'*100)
    print('MAE') 
    print(mean_absolute_error(ytest, mp))
    print('-'*100)
    print('MSE')
    print(mean_squared_error(ytest,mp))
    print('-'*100)
    print('RMSE') 
    print(np.sqrt(mean_squared_error(ytest, mp)))
    print('-'*100)
    print('R2 SCORE')
    print(r2_score(ytest,mp))

    return mp


def clean_edad(edad):   
    '''
    Función que elimina los datos de edad, que son imposibles,
    ya que le hemos dado un rango, en el cual 119 es el maximo,
    ya que es el record de longevidad.

    Args: 
        edad: columna o union de estas que contiene los datos.
    
    Return: Todas las edades reales, comprendidas en el rango impuesto.
    '''                                                  
    if edad>=0 and edad<=119:                                            
        return edad
    else:
        return np.nan

