import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pytz

# Build your `wrangle` function here
def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
    SELECT time, type, spin, 
    StyleScore, StyleValue, 
    EffectScore, EffectValue,
    SpeedScore, SpeedValue,
    stroke_counter 
    FROM motions
    """

    # Read query results into DataFrame
    # df = pd.read_sql(query, conn, index_col="time")
    df = pd.read_sql(query, conn)
    # Remove HR outliers
    # df = df[df["AVGHR"] > 50]
    # Create duration column from timestamps
    # Convert Unix timestamps to datetime objects

#    df.drop(["session_counter"])
    df = df.sort_index()  
    df = df.drop_duplicates()
    df['time'] = pd.to_datetime(df['time']/10000, unit='s')
    az_timezone = pytz.timezone('America/Phoenix')
    df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
    df['time'] = df['time'].dt.strftime('%m-%d-%Y %I:%M:%S %p')

    conn.close()
    
    return df


db_path = "/mnt/g/My Drive/Professional/BabWrangle/src/"
df = wrangle(db_path + "BabPopExt.db")
corr = df.select_dtypes("number").corr()
# sns.heatmap(corr)
summary_stats = df[["time",
                    "type",
                    "spin",
                    "StyleScore",
                    "StyleValue",
                    "EffectScore",
                    "EffectValue",
                    "SpeedScore",
                    "SpeedValue",
                    "stroke_counter"]].describe()
pd.set_option('display.max_columns', 11)
print(corr)
print(summary_stats)
# plt.hist(df["BALL_SPEED"], bins=15)

db_path2 = "/mnt/g/My Drive/Professional/BabHist/src/"
conn = sqlite3.connect(db_path2 + "OutBab.db")
df.to_sql('mytable', conn, if_exists='replace', index=True)

# Create a FacetGrid object
g = sns.FacetGrid(df, row="type", col="spin", hue="type")


# Create a list of row and column labels
row_labels = df['type'].unique()
col_labels = df['spin'].unique()


# Define a function to create a histogram
def histplot(data, **kwargs):
    sns.histplot(data=data, x="StyleScore")

# Map the histogram to the FacetGrid object
g.map_dataframe(histplot)

plt.show()

print(summary_stats)
