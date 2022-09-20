import pandas as pd
import numpy as np
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