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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn import linear_model, metrics, model_selection
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix,r2_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from datetime import datetime
from pandas_profiling import ProfileReport
=======
from skimage.io import imread
import os
import cv2
import numpy as np


## | JAVI |
sns.set_style('whitegrid')

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
    lin_reg.fit(X_train, y_train)                           #   Entrenas/generas el modelo para determinar los coeficientes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_1, random_state=random_state_1)

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

def correccion_Lasso_a_aplicar(model,X_test,y_test,X_train,y_train,alpha_1:int):
    '''
    Función para entrenar la función de Lasso y el calculo del error regularizando o sin regularizar del MSE.

    Args:
        model: modelo entrenado de regresión lineal
        X_train: Dataframe de las variables predictoras para el entrenamiento del modelo de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_train: Dataframe de las variables target para el entrenamiento del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        alpha_1:int. Número de variable alpha para entrenar la función Ridge.

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

def correccion_Lasso_a_aplicar(model, X_test, y_test, lassoR, log_ini:int,log_fin:int,n_alphas:int):
    '''
    Función que evalua la regularización de Lasso para un modelo de regresión lineal entrenado y
    que a partir de los valores logarítmicos y alpha muestra una gráfica donde se puede localizar 
    el punto más bajo de los errores y así determinar cuál es el valor de alpha más adecuado.

    Args:
        model: modelo entrenado de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal
        LassoR función de Lasso entrenada
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
        lassoR.set_params(alpha=a)
        coef_lassoR.append(lassoR.coef_)
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
        model: modelo entrenado de regresión lineal
        X_test: Dataframe de las variables predictoras para el testo del modelo de regresión lineal
        y_test: Dataframe de las variables target para el testeo del modelo de regresión lineal.

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

=======

## | LUIS | 20_09_14_28

def time_now():
    """
    Función que devuelve la fecha y hora actual
    :param: No tiene parámetros.
    :return: Tupla de strings con el día de la semana, día del mes, mes, año, hora, minuto y segundo actual.
    :rtype: tuple
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

def Feature_analisis(df):
    '''Análisis incial del df '''
    print(df.head())
    print(-*10)
    print(df.info())
    print(-*10)
    print(df.isnull().sum())
    print(-*10)
    print(df.value_counts())
    
=======

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

    Returns:
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
 
    Returns:
        Dataframe modificado.

    Ejemplo:
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

    Returns:
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

    Returns:
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

    Returns:
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

    Returns:
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

    Returns:
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

    Returns:
        Dataframe modificado
    '''

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)


        
=======
## | MARIO |

    
def read_data_bw(path, im_size, class_names_label):

    '''Lectura y etiquetado de imágenes en blanco y negro.

    Args:
        path: ruta donde estarán el resto de carpetas.

        im_size: tamaño al que queremos pasar todas las imagenes.

        class_names_label: nombre de las variables a etiquetar.
      
    Returns:
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
        path: ruta donde estarán el resto de carpetas.

        im_size: tamaño al que queremos pasar todas las imagenes.

        class_names_label: nombre de las variables a etiquetar.
      
    Returns:
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
        path: ruta donde están las imágenes.
      
    Returns:
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

    Returns: 
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

    Returns: 
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

    Returns: 
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
    
    
