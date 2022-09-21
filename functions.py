
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

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
