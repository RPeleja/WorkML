"""
    Handles loading and merging of wine and weather datasets.
    
    Methods:
        load_wine_data(red_wine_path, white_wine_path): 
            Loads and combines red and white wine datasets
            Returns: pandas.DataFrame with combined wine data
            
        load_weather_data(weather_path):
            Loads weather data and converts timestamps
            Returns: pandas.DataFrame with weather data
            
        merge_data(wine_df, weather_df):
            Merges wine and weather data based on timestamps
            Returns: pandas.DataFrame with merged data
    
    Usage:
        loader = DataLoader(config)
        wine_df = loader.load_wine_data('red.csv', 'white.csv')
        weather_df = loader.load_weather_data('weather.csv')
        merged_df = loader.merge_data(wine_df, weather_df)
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_wine_data(self, red_wine_path, white_wine_path):
        df_red = pd.read_csv(red_wine_path)
        df_red['type'] = '0'
        df_white = pd.read_csv(white_wine_path)
        df_white['type'] = '1'
        return pd.concat([df_red, df_white], ignore_index=True)

    def load_weather_data(self, weather_path):
        weather_df = pd.read_csv(weather_path, delimiter=';')
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], format='%d/%m/%Y %H:%M')
        return weather_df

    def merge_data(self, wine_df, weather_df):
        weather_filtered = weather_df[
            (weather_df['timestamp'].dt.month.isin(self.config.WEATHER_MONTHS))
        ]
        
        # Generate random dates for wine data
        min_date = weather_filtered['timestamp'].min()
        max_date = weather_filtered['timestamp'].max()
        wine_df['timestamp'] = np.random.choice(
            pd.date_range(min_date, max_date), 
            size=len(wine_df)
        )
        
        return pd.merge_asof(
            wine_df.sort_values('timestamp'),
            weather_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )