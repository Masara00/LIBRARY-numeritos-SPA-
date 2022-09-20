
def igualador(data):
    iguala = [column for column in data.columns if data[column].isna().sum() > 0]

    for column in iguala:
        data[column] = data[column].fillna(data[column].value_counts().index[0])

from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def train(model, x, y):
    model.fit(x,y)
    print(model.intercept_)
    print(model.coef_)
    mp = model.predict(x)
    print('-'*100)
    print('MAE') 
    print(mean_absolute_error(y, mp))
    print('-'*100)
    print('MSE')
    print(mean_squared_error(y,mp))
    print('-'*100)
    print('RMSE') 
    print(np.sqrt(mean_squared_error(y, mp)))
    print('-'*100)
    print('R2 SCORE')
    print(r2_score(y,mp))

    return mp