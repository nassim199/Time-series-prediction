import streamlit as st
import datetime
import numpy as np
import pandas as pd
import util
import agg_experts
from agg_experts import models

path = util.path

consommation_france = st.session_state["consommation_france"]
scores = st.session_state["scores"]
score_rte = st.session_state["score_rte"]
aggregation_experts = util.load_forecast(path + "agg_experts.csv")
st.session_state["aggregation_experts"] = aggregation_experts
score_agg = util.score(consommation_france, aggregation_experts.iloc[:, 0])
st.session_state["score_agg"] = score_agg

date = str(st.sidebar.date_input("Choisis une date", datetime.date(2022, 11, 1)))
st.header("Prediction de l'aggregation des modeles")

st.line_chart(pd.concat([ consommation_france[date:date], aggregation_experts[date:date]], axis=1))
base = min(scores.values())

methode = st.sidebar.selectbox(
    "Strategie d'aggregations",
    ("gradients de pertes", "regression avec Lasso"))
include_rte = st.sidebar.checkbox('Inclure rte')

if include_rte:
    base = min(base, score_rte)
amelioration = 100 * (base - score_agg) / base
st.write("score et amelioration par rapport au meilleur expert")
st.metric("aggregation experts", "%.2f" % score_agg, "%.2f" % amelioration + "%")


if st.sidebar.button('Aggreger les modeles'):
    try:
        if methode == "gradients de pertes":
            methode_aggregation = agg_experts.expert_agg
        else:
            methode_aggregation = agg_experts.expert_agg_lasso
        agg_experts.aggregate(methode_aggregation, include_rte).to_csv(path + "agg_experts.csv")
    except:
        st.sidebar.error("Une erreur s'est produite")