import warnings
import base64
import io
from datetime import date
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wrangle
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# input data path
file_path = "/home/blueaz/Downloads/SensorDownload/May2024/ztennis.db" 


warnings.simplefilter("ignore", UserWarning)

df = wrangle.wrangle(file_path)

# df = df.sort_values("date")
corr = df.select_dtypes("number").corr()
pd.set_option('display.max_columns', 40)
# df["client_created"] = pd.to_datetime(df["client_created"])
session = ['l_id' , 'swing_type', 'swing_side',
       'backswing_type', 'backswing_time', 'power',
        'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3',
       'dbg_gyro_1', 'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2',
       'dbg_var_3', 'dbg_var_4',  'dbg_sum_gx',
       'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax', 'dbg_max_ay',
       'dbg_min_az', 'dbg_max_az', 'impact_region',
       'ball_spin', 'impact_position_x', 'impact_position_y', 'racket_speed']

serves = ['l_id', 'impact_vel', 'ball_vel', 'spin', 'upswing_time',
         'impact_time', 'service_court']



df_session = df[session]
df_serves = df[serves]

sensor = [ 'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3', 'dbg_gyro_1',
       'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2', 'dbg_var_3', 'dbg_var_4',
       'dbg_sum_gx', 'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax',
       'dbg_max_ay', 'dbg_min_az', 'dbg_max_az']

calc = [ 'backswing_time', 'power', 'ball_spin',
        'impact_position_x', 'impact_position_y',
       'racket_speed', 'impact_region']

df_session_sensor = df_session[sensor]
df_session_calc = df_session[calc]

top_var_sensor = df_session_sensor.select_dtypes("number").var().sort_values().tail(10)
top_var_calc = df_session_calc.select_dtypes("number").var().sort_values().tail(10)
print(top_var_sensor)
print(top_var_calc)

# sns.pairplot(df_session)
# plt.show()

n_clusters = range(2,13)
inertia_errors = []
silhouette_scores = []

# Add `for` loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(df_session_calc)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    silhouette_scores.append(
            silhouette_score(df_session_calc, model.named_steps["kmeans"].labels_)
    )


# print("Inertia:", inertia_errors[:11])
# print()
# print("Silhouette Scores:", silhouette_scores[:3])

# fig = px.scatter(df, x="l_id", y="racket_speed", color="swing_type")

# plt.hist(df["ball_spin"], bins=15)
# plt.boxplot(df_session_sensor["power"], vert=False, )
# plt.show()

# fig = px.bar(
#     x=top_var_calc,
#     y=top_var_calc.index,
#     title="Zepp Tennis: High Variance Features"
# )
# 
# fig.update_layout(xaxis_title="Trimmed Var", yaxis_title="Feat")
# Create line plot of `inertia_errors` vs `n_clusters`
# fig = px.line(
#     x=n_clusters, y=inertia_errors, title="Kmeans"
# )
# fig.update_layout(xaxis_title="clust", yaxis_title="Inertia")

# fig = px.line(
#     x=n_clusters, y=silhouette_scores, title="Kmeans"
# )
# fig.update_layout(xaxis_title="clust", yaxis_title="ss")
# 
# fig = px.scatter(df_session, x="l_id", y="power", color="swing_type")

final_model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=6, random_state=42)
)

final_model.fit(df_session_calc)

labels = final_model.named_steps["kmeans"].labels_
xgb = df_session_calc.groupby(labels).mean()

# Create side-by-side bar chart of `xgb`
# fig = px.bar(
#     xgb,
#     barmode="group",
#     title="MEAN by cluster"
# )
# 
# fig.update_layout(xaxis_title = "Cluster", yaxis_title="Quantity")
# fig.show()
# Instantiate transformer
pca = PCA(n_components=2, random_state=42)

# Transform `X`
df_t = pca.fit_transform(df_session_calc)
# Put `X_t` into DataFrame
X_pca = pd.DataFrame(df_t, columns=["PC1", "PC2"])
print("X_pca shape:", X_pca.shape)
X_pca.head()

# Create scatter plot of `PC2` vs `PC1`
fig = px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title="PCA representation"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
fig.show()


print("hello")
