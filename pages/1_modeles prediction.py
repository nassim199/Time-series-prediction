import streamlit as st
import datetime
import numpy as np
import pandas as pd
import util
from agg_experts import models
import train_models

st.set_page_config(page_title="Modeles predictions France")

path = util.path
regions = util.regions

consommation_france = st.session_state["consommation_france"]

predictions_france = {}
for model in models:
    predictions_france[model] = util.load_forecast(path + f"{model}_fcast.csv").iloc[:, 0]

predictions_france["RTE"] = util.load_dataframe(path + f"consommation-france-test.txt")["Prévision J"] 

scores = {}
for model in models:
    scores[model] = util.score(consommation_france, predictions_france[model])

score_rte = util.score(consommation_france, predictions_france["RTE"])

st.session_state["scores"] = scores
st.session_state["score_rte"] = score_rte

date = str(st.sidebar.date_input("Choisis une date", datetime.date(2022, 11, 1)))
st.header("Prediction de chaque modele")
model = st.selectbox(
    'Modele de prediction',
    models  + ["RTE"])
st.line_chart(pd.concat([ consommation_france[date:date], predictions_france[model][date:date]], axis=1))

st.write("Score global de chaque modele")
cols = st.columns(len(models))
for i, model in enumerate(models):
    cols[i].metric(model, "%.2f" % scores[model])

base = min(scores.values())
amelioration = 100 * (base - score_rte) / base
st.metric("RTE", "%.2f" % score_rte, amelioration)


if st.sidebar.button('Entrainer les modeles'):
    try:
        progress_bar = st.sidebar.progress(0)
        print("Random Forest")
        train_models.train_rf().to_csv(path + "rf_fcast.csv")
        progress_bar.progress(0.10)

        print("XGBoost")
        train_models.train_xgb().to_csv(path + "xgb_fcast.csv")
        progress_bar.progress(0.20)

        print("Prophet")
        train_models.train_prophet().to_csv(path + "prophet_fcast.csv")
        progress_bar.progress(0.30)

        print("Exponential smoothing")
        train_models.train_es().to_csv(path + "es_fcast.csv")
        progress_bar.progress(0.40)

        print("SARIMAX")
        train_models.train_sarimax().to_csv(path + "sarimax_fcast.csv")
        progress_bar.progress(1)
    except:
        st.sidebar.error("Une erreur s'est produite")
st.sidebar.warning("Attention! l'entrainement des modeles va prendre plusieurs minutes...")