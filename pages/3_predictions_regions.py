import streamlit as st
import datetime
import numpy as np
import pandas as pd
import util
import region_models

regions = util.regions
path = util.path

consommations = st.session_state["consommations"]
date = str(st.sidebar.date_input("Choisis une date", datetime.date(2022, 11, 1)))

st.write("Score global pour chaque region")
prediction_region = st.selectbox(
    'Region',
    regions)

prediction_regions = {}
for region in regions:
    prediction_regions[region] = util.load_forecast(path + f"{region}-fcast.csv").iloc[:, 0]

st.header("predictions dans les regions")
st.line_chart(pd.concat([ consommations[prediction_region][date:date], prediction_regions[prediction_region][date:date]], axis=1))

scores_regions = []
for region in regions:
    error = util.score(consommations[region], prediction_regions[region])
    scores_regions.append(error)

st.session_state["scores_regions"] = scores_regions
st.table(pd.DataFrame({"region": regions, "score": scores_regions}))

if st.sidebar.button('Entrainer les modeles'):
    try:
        progress_bar = st.sidebar.progress(0)
        l = len(regions)
        for i, region in enumerate(regions):
            region_models.train_model(region)
            progress_bar.progress((i+1) / l)

        region_models.save_boundaries()
    except:
        st.sidebar.error("Une erreur s'est produite")
st.sidebar.warning("Attention! l'entrainement des modeles va prendre quelques minutes...")