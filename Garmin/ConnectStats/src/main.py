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
file_path = "/home/blueaz/Downloads/SensorDownload/Garmin/JuneSummary.csv" 


warnings.simplefilter("ignore", UserWarning)

df = wrangle.wrangle(file_path)

# df = df.sort_values("date")
corr = df.select_dtypes("number").corr()
pd.set_option('display.max_columns', 40)
# df["client_created"] = pd.to_datetime(df["client_created"])

# fig = px.scatter(df, x="l_id", y="racket_speed", color="swing_type")

# plt.hist(df["ball_spin"], bins=15)
# plt.boxplot(df_session_sensor["power"], vert=False, )
# plt.show()



fig = px.line(
    df, x='Date', y='Max HR', title="Line"
)
fig.update_layout(xaxis_title="Date", yaxis_title="ss")

fig.show()

print("complete")
