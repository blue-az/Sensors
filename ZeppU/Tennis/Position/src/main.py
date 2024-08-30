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
session = ['l_id' , 'swing_type', 'swing_side', 'hand_type',
       'backswing_type', 'backswing_time', 'power', 'stroke',
       'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3',
       'dbg_gyro_1', 'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2',
       'dbg_var_3', 'dbg_var_4',  'dbg_sum_gx',
       'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax', 'dbg_max_ay',
       'dbg_min_az', 'dbg_max_az', 'impact_region', 'diffxy',
       'ball_spin', 'impact_position_x', 'impact_position_y', 'racket_speed']

serves = ['l_id', 'impact_vel', 'ball_vel', 'spin', 'upswing_time',
         'impact_time', 'service_court']



df_session = df[session]
df_serves = df[serves]

sensor = [ 'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3', 'dbg_gyro_1',
       'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2', 'dbg_var_4',
       'dbg_sum_gx', 'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax',
          'dbg_max_ay', 'dbg_min_az', 'dbg_max_az', 'hand_type']

calc = [ 'backswing_time', 'diffxy', 'ball_spin',
        'impact_position_x', 'impact_position_y',
       'racket_speed', 'impact_region', 'hand_type', 'stroke']

df_session_sensor = df_session[sensor]
df_session_calc = df_session[calc]

corr_calc = df_session_calc.select_dtypes("number").corr()
corr_sensor = df_session_sensor.select_dtypes("number").corr()

# sns.pairplot(df_session_calc, hue='stroke')
# plt.show()

# Create a FacetGrid object for each is_hit_frame value
# g = sns.FacetGrid(df, row="swing_side", col="swing_type")

# Map the scatter plot to the FacetGrid object
# g.map(sns.stripplot, "impact_position_x", "impact_position_y", jitter=True)

# Remove the hue variable
# g2.hue_var = None


# plt.hist(df["stroke"], bins=50)
# plt.boxplot(df_session_sensor["power"], vert=False, )
# plt.show()

# fig = px.bar(
#     x=top_var_calc,
#     y=top_var_calc.index,
#     title="Zepp Tennis: High Variance Features"
# )
# 
# fig.update_layout(xaxis_title="Trimmed Var", yaxis_title="Feat")

fig = px.scatter(df_session, x="backswing_time", y="racket_speed", color='impact_region')
# fig = px.scatter_3d(df_session, x="impact_position_x", y="impact_position_y", z="power", color="stroke")

fig.show()


print("hello")
