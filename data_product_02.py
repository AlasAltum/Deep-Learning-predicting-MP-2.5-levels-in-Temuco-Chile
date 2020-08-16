"""
This script is for generating the 'data_product_02.pk'
"""
from preprocessing import open_df, multi_date, process_mp25
from functools import reduce
import pandas as pd
import pickle
from datetime import datetime
import numpy as np


# opening pandas
wind_dir = open_df('./data/dir_viento.csv', 'wind_dir')
wind_vel = open_df('./data/vel_viento.csv', 'wind_vel')
p_atm = open_df('./data/p_atm.csv', 'p_atm')
rel_hum = open_df('./data/humedad_relativa.csv', 'rel_hum')
precipitations = open_df('./data/precipitaciones.csv', 'precipitations')
temp = open_df('./data/temperatura_ambiente.csv', 'temp')
mp25 = process_mp25('./data/mp25.csv', 'mp_25')

# combining dataframes into one
data_frames = [wind_dir, wind_vel, p_atm, rel_hum, precipitations, temp, mp25]
df_merged = reduce(lambda left, right: pd.merge(left, right,
                                                on=['datetime'],
                                                how='outer'), data_frames)
#making multi index
df_merged = multi_date(df_merged)
df_merged.drop('p_atm', axis=1, inplace=True)

# making data product 2
df_merged.set_index('datetime', drop=True, inplace=True)
df_merged = df_merged.resample('2H').mean()
df_merged = df_merged.interpolate('spline', order=3).dropna()
df_merged = df_merged[df_merged.index < datetime(year=2019, month=5, day=15)]
df_merged['mp_25'][df_merged['mp_25'] < 0] = 0

t = np.linspace(0, 1000, len(df_merged['mp_25']))

y = 290 * np.sin(t / 31) + 320
yy = 3.3 - .009 * t
yyy = -np.abs(30 * np.sin(t / 60 + 200)) + 40
yyyy = t * 0 + 12

df_merged['mp_25'][df_merged['mp_25'] > y] = y
df_merged['temp'][df_merged['temp'] < yy] = yy
df_merged['rel_hum'][df_merged['rel_hum'] < yyy] = yyy
df_merged['wind_vel'][df_merged['wind_vel'] > yyyy] = yyyy
df_merged['wind_vel'][df_merged['wind_vel'] < 0] = 0

# saving with pickle
with open('./data/data_product_02.pk', 'wb') as f:
    pickle.dump(df_merged, f)
f.close()
