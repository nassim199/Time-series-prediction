from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from prophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import util

path = util.path
train_path = path+'consommation-france.txt'
X, Consommation = util.construct_features(train_path, regions=['temp_mean'])

test_path = path+'consommation-france-test.txt'
X_test, Consommation_test = util.construct_features(test_path, regions=['temp_mean'])

def train_rf():
    regr = RandomForestRegressor()
    regr.fit(X, Consommation)

    forecast = regr.predict(X_test)
    forecast = pd.Series(forecast).set_axis(Consommation_test.index)
    error = util.score(Consommation_test , forecast)
    print("score with Random Forest : ", error)
    return forecast

def train_xgb():
    param = {'max_depth': 5, 'eta': 1, 'objective': 'reg:squarederror'}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'

    dtrain = xgb.DMatrix(X, label=Consommation)

    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    dtest = xgb.DMatrix(X_test)
    forecast = bst.predict(dtest)
    forecast = pd.Series(forecast).set_axis(Consommation_test.index)
    error = util.score(Consommation_test , forecast)
    print("score with XGBoost : ", error)
    return forecast


def train_prophet():
    prophet = Prophet()
    df = util.load_dataframe(train_path)
    Consommation = df.Consommation
    ts = pd.DataFrame({'ds':Consommation.index,'y':Consommation})
    prophet.fit(ts)

    df = util.load_dataframe(test_path)
    Consommation_test = df.Consommation
    # create a future data frame 
    future = pd.DataFrame({'ds':Consommation_test.index})
    forecast = prophet.predict(future)

    # display the most critical output columns from the forecast
    forecast = forecast[['ds','yhat','yhat_lower','yhat_upper']]
    forecast.set_index('ds', inplace=True)
    ts_pred = forecast.yhat

    error = util.score(Consommation_test , ts_pred)
    print("score with Prophet : ", error)
    return ts_pred


def train_es():
    df = util.load_dataframe(train_path)
    Consommation = df.Consommation['2022-04-01':]
    df = util.load_dataframe(test_path)
    Consommation_test = df.Consommation


    fit = ExponentialSmoothing(
        Consommation,
        seasonal_periods=48,
        trend="add",
        seasonal="add",
        initialization_method="estimated",
    ).fit()


    fcast = fit.forecast(48).rename("Additive")
    total_fcast = pd.Series([])


    while len(Consommation_test) > 0:
        total_fcast = pd.concat([total_fcast, fcast])
        Consommation = pd.concat([Consommation[48:], Consommation_test[fcast.index]])
        Consommation_test = Consommation_test.drop(fcast.index)
        fit = ExponentialSmoothing(
            Consommation,
            seasonal_periods=48,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        ).fit()

        m = min(48, len(Consommation_test))
        if m > 0:
            fcast = fit.forecast(m)
    

    Consommation_test = df.Consommation
    error = util.score(total_fcast, Consommation_test)
    print("score with Exponential Smoothing : ", error)
    return total_fcast


def train_sarimax():
    df = util.load_dataframe(train_path)
    Consommation = df.Consommation['2022-05-01':]
    df = util.load_dataframe(test_path)
    Consommation_test = df.Consommation

    model = SARIMAX(Consommation, order=(0,0,0), seasonal_order=(0,1,1,48))
    model_fit = model.fit()
    fcast = model_fit.forecast(steps=48)
    total_fcast = pd.Series([])


    while len(Consommation_test) > 0:
        total_fcast = pd.concat([total_fcast, fcast])
        date = fcast.index[0]

        print(date)

        date = date + pd.DateOffset(1)
        
        if date.day == 1:
            date = date - pd.DateOffset(months=1)
            Consommation = df.Consommation[str(date.date()):str(fcast.index[0].date())]
            Consommation_test = Consommation_test.drop(fcast.index)
            model = SARIMAX(Consommation, order=(0,0,0), seasonal_order=(0,1,1,48))
            model_fit = model.fit()
            fcast = model_fit.forecast(steps=48)
        else:
            model_fit = model_fit.append(Consommation_test[fcast.index], refit=False)
            Consommation_test = Consommation_test.drop(fcast.index)
            m = min(48, len(Consommation_test))
            if m > 0:
                fcast = model_fit.forecast(steps=m)

    Consommation_test = df.Consommation
    error = util.score(total_fcast, Consommation_test)
    print("score with SARIMAX : ", error)
    return total_fcast

