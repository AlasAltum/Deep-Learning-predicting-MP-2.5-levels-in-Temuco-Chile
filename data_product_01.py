"""
This script is for generating the 'data_product_01.pk'
"""
from preprocessing import open_df, multi_date, process_mp25
from functools import reduce
import pandas as pd
import pickle

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
df_merged = reduce(lambda  left, right: pd.merge(left, right,
                                                 on=['datetime'],
                                                 how='outer'), data_frames)
#making multi index
df_merged = multi_date(df_merged)

# saving with pickle
with open('./data/data_product_01.pk', 'wb') as f:
    pickle.dump(df_merged, f)
f.close()
