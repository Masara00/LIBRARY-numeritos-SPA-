'''
Librerias a utilizar
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet,LogisticRegression,metrics, model_selection
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,VotingRegressor,ExtraTreesRegressor,RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.svm import SVR,SVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score,confusion_matrix,r2_score
from sklearn import preprocessing

sns.set_style('whitegrid')

def graficas(df:dataframe,y):
    '''
    Función para representar varias graficas antes de realizar cualquier modelo
    df es un DataFrame
    y es la variable dependiente y se expresa como df(['y'])
    '''
    plt.figure(figsize=(20,20))
    sns.pairplot(df)
    fig, axes = plt.subplots(2,1)
    sns.distplot(y, ax = axes[0])
    sns.heatmap(df.corr(), annot=True, ax = axes[1])
    axes[0].set_title("Distribucion")
    axes[1].set_title("Mapa Correlación");

def funcion_lineal_regression(X:list,y,test_size_1:float,random_state_1:int):
    '''Función para ingresar los datos de las variables previsoras (X) y la variable target (y), 
    los parámetros necesarios para realizar el train test split (random_state y test_size).

    En esta función se extrae como salida el módelo entrenado, el coeff_df, los valores de intercepto y las gradientes. 
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

    return lin_reg,coeff_df, lin_reg.intercept_,lin_reg.coef_

def función_metricas_error (model:function,X_test:Dataframe,y_test:DataFrame,X_train:DataFrame,y_train:DataFrame):
    '''
    Función que a partir de la función entrenada te facilita las métricas más importantes en regresión lineal.
    Se ingresa la función entrenada y los valores de test: X_test,y_test.
    También se ingresan los datos de entrenamiento del  módelo para calcular las métricas de error

    Se extrae de esta función las variables:
    mae_pred,mape_pred,mse_pred,msqe_pred,mae_train,mape_train,mse_train,msqe_train
    OJO!! Tengo dudas sobre si el model se queda entrenado.
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

def funcion_ridge (model:function,X_test:Dataframe,y_test:DataFrame,X_train:DataFrame,y_train:DataFramme,alpha_1:int):
    '''Función para entrenar la función de ridge y el calculo del error regularizando o sin regularizar del MSE.
    Se extrae de esta función la función de Ridge entrenada.'''
    
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

def correccion_Lasso_a_aplicar(model:function,X_test:Dataframe,y_test:DataFrame,X_train:DataFrame,y_train:DataFramme,alpha_1:int):
    '''Función para entrenar la función de Lasso y el calculo del error regularizando o sin regularizar del MSE.
    Se extrae de esta función la función de Ridge entrenada.'''


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

def correccion_ridge_a_aplicar(model:function, ridgeR:function, X_train:Dataframe, X_test:Dataframe, y_train:Dataframe, y_test:Dataframe,log_ini:int,log_fin:int,n_alphas:int):
    '''Función que evalua la regularización de Ridge para un modelo de regresión lineal entrenado y
    que a partir de los valores logarítmicos y alpha muestra una gráfica donde se puede localizar 
    el punto más bajo de los errores y así determinar cuál es el valor de alpha más adecuado.
    
    OJO!!! esta función está por revisar'''
    predictions = model.predict(X_test)                   #   Determino los resultados que deberían de dar con los valores guardados para

    alphas = np.logspace(log_ini, log_fin, n_alphas) 
    baseline_error = metrics.mean_squared_error(y_test, predictions)
    coef_ridge = []
    err_ridge = []
    baseline = []

    for a in alphas:
        ridge = Ridge(alpha=a)
        ridge.fit(X_train, y_train)
        
        coef_ridge.append(ridge.coef_)
        
        y_pred = ridge.predict(X_test)
        ridge_error = metrics.mean_squared_error(y_pred, y_test)
        
        err_ridge.append(ridge_error)
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

def correccion_Lasso_a_aplicar(model:function, lassoR:function, X_train:DataFrame, X_test:DataFrame, y_train:DataFrame, y_test:DataFrame,log_ini:int,log_fin:int,n_alphas:int):
    '''Función que evalua la regularización de Lasso para un modelo de regresión lineal entrenado y
    que a partir de los valores logarítmicos y alpha muestra una gráfica donde se puede localizar 
    el punto más bajo de los errores y así determinar cuál es el valor de alpha más adecuado.
    
    OJO!!! esta función está por revisar'''
    predictions = model.predict(X_test)
    alphas = np.logspace(log_ini, log_fin, n_alphas) 
    baseline_error = metrics.mean_squared_error(y_test, predictions)
    coef_ridge = []
    err_ridge = []
    baseline = []

    for a in alphas:
        lassoR.set_params(alpha=a)
        lassoR.fit(X_train, y_train)
        coef_lassoR.append(lassoR.coef_)
        y_pred = lassoR.predict(X_test)
        lasso_error = metrics.mean_squared_error(y_pred, y_test)    
        err_lasso.append(lasso_error)

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

def error_modelo(model:function, X_test:DataFrame, y_test:DataFrame):
    '''
    Función que a partir de un modelo entrenado con las variables X_test e y_test, muestra las
    métricas más relevantes de un módelo clasificatorio y devuelve un DataFrame de los mismos.
    '''
    y_pred = model.predict(X_test)
    f1_model=f1_score(y_test, y_pred,average='macro')
    acc_model=accuracy_score(y_test, y_pred)
    precision_model=precision_score(y_test, y_pred,average='macro')
    recall_model=recall_score(y_test, y_pred,average='macro')
    roc_auc_model=roc_auc_score(y_test, model.predict_proba(X_test),multi_class='ovr')
    conf_model=confusion_matrix(y_test, y_pred, normalize='true')
    model_error = {'accuracy': acc_model, 'f-1': f1_model, 'recall': recall_model , 'precision': precision_model}
    df=pd.DataFrame.from_dict(model_error,orient='index')

    print('Accuracy', acc_model)
    print('F1', f1_model)
    print('Precision', precision_model)
    print('Recall', recall_model)
    print('-'*30)
    print('ROC', roc_auc_model)

    plt.figure(figsize=(10,10))
    sns.heatmap(conf_model, annot=True)
    return df

def PruebaModelos(x, y, xtest, ytest, ModelosRegresion = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), DecisionTreeRegressor(), RandomForestRegressor(), ExtraTreesRegressor(), KNeighborsRegressor(), SVR()], 
ModelosClasificacion = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), ExtraTreesClassifier(), KNeighborsClassifier(), SVC()], 
agregar = [], quitar = [], metricas = [], tipo = "regresion"):

    medidas = []
    resultado = ""
    if tipo == "regresion":
        for i in agregar:
            ModelosRegresion.append(i)
        for i in quitar:
            ModelosRegresion.remove(i)
        for modelo in ModelosRegresion:
            if modelo != SVR():
                modelo.fit(x, y)
            else:
                stdr = StandardScaler.fit_transform(x)
                modelo.fit(stdr, y)
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
                modelo.fit(x, y)
            else:
                stdr = StandardScaler.fit_transform(x)
                modelo.fit(stdr, y)
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
    if max == None:
        resultado = data.corr()[(data.corr() > min) & (data.corr() != 1)].dropna(axis = 1, how = "all").dropna(axis = 0, how = "all")
    else:
        resultado = data.corr()[(data.corr() > min) & (data.corr() < max)].dropna(axis = 1, how = "all").dropna(axis = 0, how = "all")
    return resultado

def root_mean_squared_error(y_true, y_pred):
    return np.square(metrics.mean_squared_error(y_true, y_pred))

def DfObjNum(data, type1 = "object", type2 = "float64"):
    for i in data.dtypes[data.dtypes == type1].index:
        data[i] = data[i].astype(type2)