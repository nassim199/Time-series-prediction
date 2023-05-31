from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import util
import numpy as np

import json

path = util.path

regions = util.regions
region_boundaries = {}

def train_model(region):
    X, Consommation = util.construct_features(path+f'consommation-{region}.txt', regions=[region])
    regr = RandomForestRegressor()
    regr.fit(X, Consommation)

    X_test, Consommation_test = util.construct_features(path + f"consommation-{region}-test.txt", regions=[region])

    train_forecast = regr.predict(X)
    error = train_forecast - Consommation
    region_boundaries[region] = [np.quantile(np.abs(error), q) for q in np.arange(1,10)/10]
    forecast = regr.predict(X_test)
    forecast = pd.Series(forecast).set_axis(Consommation_test.index)

    error = util.score(Consommation_test , forecast)
    print(f"score with Random Forest ({region}): ", error)
    forecast.to_csv(path + f"{region}-fcast.csv")


def save_boundaries():
    with open(path + 'boundaries.txt', 'w') as boundaries:
        boundaries.write(json.dumps(region_boundaries))
