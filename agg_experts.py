import pandas as pd
import numpy as np
import util 

path = util.path
df = util.load_dataframe(path+'consommation-france-test.txt')
Consommation_test = df.Consommation

models = ["rf", "xgb", "prophet", "es", "sarimax"]
experts = {}
for model in models:
  experts[model] = util.load_forecast(path+f"{model}_fcast.csv")



from scipy.special import softmax
from sklearn import linear_model

def expert_agg(past_agg, past_preds, past_y, experts, windowing=None):
  if windowing is not None:
    past_agg = past_agg[-windowing:]
    past_preds = past_preds[-windowing:]
    past_y = past_y[-windowing:]
  M = np.max((past_preds - past_y[:, None])**2)
  N = experts.shape[1]
  T = experts.shape[0]
  eta = np.sqrt( 8*np.log(N) / T)/M
  diff = 2 * (past_agg - past_y)
  l = past_preds * diff[:, None]
  p = -eta * l.sum(axis=0)
  p = softmax(p)
  agg = experts @ p
  return agg

def expert_agg_lasso(past_agg, past_preds, past_y, experts, windowing=None):
  if windowing is not None:
    past_agg = past_agg[-windowing:]
    past_preds = past_preds[-windowing:]
    past_y = past_y[-windowing:]

  reg = linear_model.Lasso(alpha=0.1).fit(past_preds, past_y)
  agg = reg.predict(experts)
  return agg


def aggregate(agg_method, include_rte=False):
    #on peut inclure les predictions de RTE dans notre modele
    #df["Prévision J"]
    if include_rte:
        df_experts = pd.concat([experts[model][f"{model}_fcast"] for model in models] + [Consommation_test, df["Prévision J"]], axis=1)
    else:
        df_experts = pd.concat([experts[model][f"{model}_fcast"] for model in models] + [Consommation_test], axis=1)
    df_experts.dropna(inplace=True)

    y = df_experts.Consommation
    df_experts.drop("Consommation", axis=1, inplace=True)

    dates = pd.Series(y.index).dt.date.astype(str).unique()
    d = dates[0]
    agg_experts = df_experts[d:d].mean(axis=1)
    past_preds = df_experts[:d]
    past_y = y[:d]

    for d in dates[1:]:
        experts_ = df_experts[d:d]
        agg_experts_ = agg_method(agg_experts.values, past_preds.values, past_y.values, experts_.values)
        agg_experts_ = pd.Series(agg_experts_).set_axis(y[d:d].index)

        past_y = y[:d]
        agg_experts = pd.concat([agg_experts, agg_experts_])
        past_preds = df_experts[:d]

    error = util.score(agg_experts, y)
    print("error of experts aggregates : ", error)
    return agg_experts


