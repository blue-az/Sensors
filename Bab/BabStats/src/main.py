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
# from scipy.stats.mstats import trimmed_var
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# PC path
file_path = "/mnt/g/My Drive/Professional/Bab/BabWrangle/src/BabPopExt.db"
# file_path = "/home/efehn2000/GoogHome/Professional/BabWrangle/src/BabPopExt.db"


warnings.simplefilter("ignore", UserWarning)

df = wrangle.wrangle(file_path)
df.shape
df["time"] = pd.to_datetime(df["time"])

dfna = df.dropna()

df.info()
# df = df.sort_values("date")
corr = df.select_dtypes("number").corr()

summary_stats = df[["time",
                    "type",
                    "spin",
                    "StyleScore",
                    "StyleValue",
                    "EffectScore",
                    "EffectValue",
                    "SpeedScore",
                    "SpeedValue",
                    "stroke_counter",
                    "PIQ"]].describe()
pd.set_option('display.max_columns', 11)
print(corr)
print(summary_stats)

# sns.pairplot(dfna, hue="type")
# plt.show()

# plt.hist(df["PIQ"], bins=15)
# plt.boxplot(dfna["EffectScore"], vert=True)
# plt.show()

# fig = px.scatter(df, x="time", y="PIQ", trendline="ols")
fig = px.scatter(df, x="time", y="PIQ", color="spin")
fig.show()

print("hello")
