import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

path = "./data/"

def score(test, pred):
  return np.sqrt(np.mean(np.square(test - pred)))



jrs_feries = pd.read_csv(path+"jours_feries_metropole.csv").date.values

regions = ["Auvergne-Rhône-Alpes", "Bourgogne-Franche-Comté", "Bretagne", "Centre-Val-de-Loire", "Grand-Est", "Hauts-de-France", "Ile-de-France", "Normandie", "Nouvelle-Aquitaine", "Occitanie", "PACA", "Pays-de-la-Loire"]
regions_stations = {
    "Auvergne-Rhône-Alpes": ["7460", "7481", "7577"], 
    "Bourgogne-Franche-Comté": ["7280"], 
    "Bretagne": ["7110", "7117", "7130", "7207"], 
    "Centre-Val-de-Loire": ["7255", "7240"], 
    "Grand-Est": ["7072", "7168", "7181", "7190", "7299"], 
    "Hauts-de-France": ["7005", "7015"], 
    "Ile-de-France": ["7149"], 
    "Normandie": ["7020", "7027", "7037", "7139"], 
    "Nouvelle-Aquitaine": ["7314", "7335", "7434", "7510", "7607"], 
    "Occitanie": ["7535", "7558", "7621", "7627", "7630", "7643", "7650"], 
    "PACA": ["7591", "7690"], 
    "Pays-de-la-Loire": ["7222"]
}

temperature_df = pd.read_csv(path+"temperature.csv", index_col = 'date', parse_dates=['date'])
temperature_df.drop("7661", axis=1, inplace=True)
temperature_df = temperature_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
temperature_df['temp_mean'] = temperature_df.mean(axis=1)

for region in regions:
  temperature_df[region] = temperature_df[regions_stations[region]].mean(axis=1)

global_dataframes = {}
def load_dataframe(file_path, force_reload=False):
  if force_reload or file_path not in global_dataframes:
    df = pd.read_csv(file_path, sep='\t', encoding='latin-1', usecols=range(7), index_col=False)
    df.drop(df.tail(1).index,inplace=True)

    #df = df[df['Consommation'].notna()]
    df = df[df['Heures'].str[-1] != '5']
    df['Consommation'] = pd.to_numeric(df['Consommation'], errors='coerce').interpolate(limit_area='inside')
    df = df[df['Consommation'].notna()]
    
    #df.reset_index(drop=True, inplace=True)
    df["time"] = df.apply(lambda row: datetime.strptime(row["Date"] + " " + row["Heures"] , '%Y-%m-%d %H:%M'), axis=1)
    df.set_index('time', inplace=True)
    global_dataframes[file_path] = df
    return df
  else:
    return global_dataframes[file_path]

def construct_features(file_path, regions=regions):
    df = load_dataframe(file_path)
    Consommation = df["Consommation"]
    X = pd.DataFrame(Consommation)
    X["weekday"] = pd.Series([d.weekday() for d in Consommation.index]).set_axis(Consommation.index)
    X["period"] = pd.Series([int(d.hour*2 + d.minute/30) for d in Consommation.index]).set_axis(Consommation.index)
    X["Consoomation_48"] = X["Consommation"].shift(48)
    X["Consoomation_336"] = X["Consommation"].shift(336)
    X.dropna(inplace=True)

    X = pd.merge(X, temperature_df[regions], how="left", left_index=True, right_index=True)
    for col in regions:
        X[col] = X[col].astype(float).interpolate()
    X['jr_ferie'] = pd.Series(X.index).dt.date.astype(str).isin(jrs_feries).set_axis(X.index)


    Consommation = X.Consommation
    X.drop("Consommation", axis=1, inplace=True)
    return X, Consommation

def load_forecast(file_path):
  df = pd.read_csv(file_path, index_col=0, parse_dates=[0])
  df.columns = [file_path.split("/")[-1][:-4]]
  return df