import sqlite3
import pandas as pd
import pytz

# Build your `wrangle` function here
def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
    SELECT * 
    FROM swings
    """
    # Read query results into DataFrame
    # df = pd.read_sql(query, conn, index_col="time")
    df = pd.read_sql(query, conn)
    df = df.sort_index()  
    # df = df.drop_duplicates()
    
    df['l_id'] = pd.to_datetime(df['l_id'], unit='ms')
    az_timezone = pytz.timezone('America/Phoenix')
    df['l_id'] = df['l_id'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
    df['l_id'] = df['l_id'].dt.strftime('%m-%d-%Y %I:%M:%S %p')
    df["l_id"] = pd.to_datetime(df["l_id"])

    df.dropna(inplace=True)
    df = df.sort_values("l_id")
    # df.set_index('l_id', inplace=True)
    # Replace # with descriptions
    hand_type = {2: "BH", 1: "FH"}
    swing_type = {4: "VOLLEY", 3: "SERVE", 2: "TOPSPIN", 0: "SLICE", 1: "FLAT", 5: "SMASH"}
    df['swing_type'] = df['swing_type'].replace(swing_type)
    df['hand_type'] = df['swing_side'].replace(hand_type)

    conn.close()
    
    return df
