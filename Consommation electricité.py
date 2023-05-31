import streamlit as st
import pandas as pd
import numpy as np
import datetime 
import util
import download_data
from agg_experts import models

path = util.path
regions = util.regions

consommations = {}
consommations['france'] = util.load_dataframe(path + f"consommation-france-test.txt").Consommation
for region in regions:
    consommations[region] = util.load_dataframe(path + f"consommation-{region}-test.txt").Consommation
    
st.session_state["consommations"] = consommations


st.set_page_config(
    page_title="Données Consommation d'energie"
)
st.title("Consommation d'electricité")

date = str(st.sidebar.date_input("Choisis une date", datetime.date(2022, 11, 1)))

st.header("Consommation sans prediction")
region = st.selectbox(
    'Etendu (France où region)',
    ['france'] + regions)

st.line_chart(consommations[region][date:date])
if st.sidebar.button('Download data'):
    try:
        download_data.dowload_data()
    except:
        st.sidebar.error("Une erreur dans le telechargement s'est produite")
st.sidebar.info("le telechargement automatique ne marche que sur machine linux")

consommation_france = consommations['france']
st.session_state["consommation_france"] = consommation_france

