import os
import pandas as pd
from datetime import datetime
import util

regions = util.regions
path = util.path

def dowload_data():
  os.system("wget https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-Consolide.zip")
  os.system("unzip eCO2mix_RTE_En-cours-Consolide.zip")
  os.system(f"mv eCO2mix_RTE_En-cours-Consolide.xls {path}consommation-france.txt")
  os.system("rm eCO2mix_RTE_En-cours-Consolide.zip")

  os.system("wget https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_En-cours-TR.zip")
  os.system("unzip eCO2mix_RTE_En-cours-TR.zip")
  os.system(f"mv eCO2mix_RTE_En-cours-TR.xls {path}consommation-france-test.txt")
  os.system("rm eCO2mix_RTE_En-cours-TR.zip")

  os.system("wget https://www.data.gouv.fr/fr/datasets/r/6637991e-c4d8-4cd6-854e-ce33c5ab49d5")
  os.system(f"mv 6637991e-c4d8-4cd6-854e-ce33c5ab49d5 {path}jours_feries_metropole.csv")


  months = [str(month)[:7].replace('-', '') for month in pd.date_range(start="2021-01",end="2023-01", freq='M')]

  for date in months:
    download = f'wget https://donneespubliques.meteofrance.fr/donnees_libres/Txt/Synop/Archive/synop.{date}.csv.gz'
    os.system(download)

    unzip = f'gunzip synop.{date}.csv.gz'
    os.system(unzip)

    file_name = f'synop.{date}.csv'

    meteo_df = pd.read_csv(file_name, sep=';')
    meteo_df = meteo_df[["numer_sta", "date", "t"]]
    #filtrer que les stations en france
    meteo_df = meteo_df[meteo_df["numer_sta"] <= 7690]
    meteo_df["date"] = meteo_df["date"].apply(lambda d:datetime.strptime(str(d), '%Y%m%d%H%M%S'))
    meteo_df = meteo_df.set_index(["numer_sta", "date"]).unstack("numer_sta")
    meteo_df = meteo_df.t.rename_axis(None,axis=1)
    first_month = (date == months[0])
    meteo_df.to_csv("temperature.csv", mode='w' if first_month else 'a', header=first_month)

    delete_file = f'rm {file_name}'
    os.system(delete_file)
  
  os.system(f"mv temperature.csv {path}temperature.csv")

  #telechargement des données de consommation pour les autres regions


  for region in regions:
    download = f'wget https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{region}_En-cours-Consolide.zip'
    os.system(download)

    unzip = f"unzip -p eCO2mix_RTE_{region}_En-cours-Consolide.zip > {path}consommation-{region}.txt"
    os.system(unzip)

    delete_file = f'rm eCO2mix_RTE_{region}_En-cours-Consolide.zip'
    os.system(delete_file)

    if region == "Auvergne-Rhône-Alpes":
      region_ = "Auvergne-Rhone-Alpes"
    elif region == "Bourgogne-Franche-Comté":
      region_ = "Bourgogne-Franche-Comte"
    else:
      region_ = region
    download = f'wget https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_{region_}_En-cours-TR.zip'
    os.system(download)

    unzip = f"unzip -p eCO2mix_RTE_{region_}_En-cours-TR.zip > {path}consommation-{region}-test.txt"
    os.system(unzip)

    delete_file = f'rm eCO2mix_RTE_{region}_En-cours-TR.zip'
    os.system(delete_file)
