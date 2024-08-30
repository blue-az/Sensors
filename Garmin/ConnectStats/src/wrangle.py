import pandas as pd
import pytz

# Build your `wrangle` function here
def wrangle(file_path):
    # Connect to csv file
    df = pd.read_csv(file_path)
    
    # df['l_id'] = pd.to_datetime(df['l_id'], unit='ms')
    # az_timezone = pytz.timezone('America/Phoenix')
    # df['l_id'] = df['l_id'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
    # df['l_id'] = df['l_id'].dt.strftime('%m-%d-%Y %I:%M:%S %p')
    # df["l_id"] = pd.to_datetime(df["l_id"])
    mask = ['Activity Type', 'Date', 'Favorite', 'Title', 'Distance', 'Calories',
       'Total Time', 'Avg HR', 'Max HR', 'Avg Bike Cadence',
       'Max Bike Cadence', 'Avg Speed', 'Max Speed', 'Avg Stride Length',
       'Elapsed Time']
    df = df[mask]
    
    return df
