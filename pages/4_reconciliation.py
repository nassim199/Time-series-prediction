import streamlit as st
import datetime
import numpy as np
import pandas as pd
import util
import reconciliation

regions = util.regions
path = util.path

consommations = st.session_state["consommations"]
consommation_france = st.session_state["consommation_france"]
scores_regions = st.session_state["scores_regions"]
score_agg = st.session_state["score_agg"]

date = str(st.sidebar.date_input("Choisis une date", datetime.date(2022, 11, 1)))

st.header("predictions aprés reconciliation")
predictions_reconciliation = {}
predictions_reconciliation['france'] = util.load_forecast(path + "total_reconciliated.csv").iloc[:, 0]
for region in regions:
    predictions_reconciliation[region] = util.load_forecast(path + f"{region}-reconciliated.csv").iloc[:, 0]

region_reconciliation = st.selectbox(
    'Etendu (France où region) de la reconciliation',
    ['france'] + regions)

st.line_chart(pd.concat([ consommations[region_reconciliation][date:date], predictions_reconciliation[region_reconciliation][date:date]], axis=1))

score_reconciliation = util.score(consommation_france, predictions_reconciliation["france"])
base = score_agg
amelioration = 100 * (base - score_reconciliation) / base
st.write("score et amelioration par rapport au score avant reconciliation")
st.metric("France ", "%.2f" % score_reconciliation, "%.2f" % amelioration + "%")
st.write("pour les regions:")

scores_reconciliation = []
ameliorations_reconciliation = []
for i, region in enumerate(regions):
    error = util.score(consommations[region], predictions_reconciliation[region])
    scores_reconciliation.append(error)
    base = scores_regions[i]
    amelioration_reconciliation = 100 * (base - error) / base
    ameliorations_reconciliation.append(amelioration_reconciliation)

def color(val):
    color = 'green' if val >= 0 else 'red'
    return f'color: {color}'

st.table(pd.DataFrame({"region": regions, "score": scores_regions, "score aprés reconciliation":scores_reconciliation, "amelioration (en %)": ameliorations_reconciliation}).style.applymap(color, subset=['amelioration (en %)']))

methode = st.sidebar.selectbox(
    "Strategie de reconciliation",
    ("OLS", "Bottom-up", "Gtop"))
quantile = None
if methode == 'Gtop':
    quantile = st.sidebar.select_slider('Quantille', options=range(10, 100, 10))
if st.sidebar.button('Reconcilier les resultats'):
    try:
        
        if methode == "OLS":
            methode_reconciliation = reconciliation.ols_reconciliation
        elif methode == "Bottom-up":
            methode_reconciliation = reconciliation.bottom_up_reconciliation
        else:
            methode_reconciliation = reconciliation.gtop_reconciliation
            
            st.sidebar.warning("Attention! la reconciliation des modeles peut prendre quelques minutes...")

        new_gb_forecast, region_corrected = reconciliation.reconciliate(methode_reconciliation, quantile)

        new_gb_forecast.to_csv(path+ 'total_reconciliated.csv')
        for region in region_corrected:
            region_corrected[region].to_csv(path + f"{region}-reconciliated.csv")
    except:
        st.sidebar.error("Une erreur s'est produite")