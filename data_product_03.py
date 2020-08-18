import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import *
import sys
import pickle
from preprocessing import get_consecutive

length, amount = int(sys.argv[1]), sys.argv[2]


def get_slices(df, amount):
    """
    Esta función toma un df y lo separa cada vez que pilla que el index,
    que tiene que ser un datetime, tiene un intervalo distinto al indicado
    """
    indice = df.index
    diferencia = indice.to_series().diff()
    indices_a_dividir = np.where(np.logical_not(diferencia > amount))[0].tolist()
    groups = []
    for k, g in groupby(enumerate(indices_a_dividir), lambda x: x[0] - x[1]):
        groups.append(list(map(itemgetter(1), g)))
    dfs = []
    for idx in groups:
        dfs.append(df.iloc[idx])

    return dfs


# opening data
main_path = './data/cleaned-data/'
padre_las_casas = pd.read_csv(main_path + 'padre_las_casas_ii_cleaned_data.csv')
las_encinas = pd.read_csv(main_path + 'las_encinas_cleaned_data.csv')
nielol = pd.read_csv(main_path + 'ñielol_cleaned_data.csv')

# making datetime the new index
padre_las_casas.set_index('datetime', inplace=True, drop=True)
las_encinas.set_index('datetime', inplace=True, drop=True)
nielol.set_index('datetime', inplace=True, drop=True)

padre_las_casas.dropna(inplace=True)
las_encinas.dropna(inplace=True)
nielol.dropna(inplace=True)

padre_las_casas.index = pd.to_datetime(padre_las_casas.index)
las_encinas.index = pd.to_datetime(las_encinas.index)
nielol.index = pd.to_datetime(nielol.index)

padre_las_casas.drop('Estacion', axis=1, inplace=True)
las_encinas.drop('Estacion', axis=1, inplace=True)
nielol.drop('Estacion', axis=1, inplace=True)

mp_25 = {'mp_2,5': 'mp_25'}
padre_las_casas.rename(columns=mp_25, inplace=True)
las_encinas.rename(columns=mp_25, inplace=True)
nielol.rename(columns=mp_25, inplace=True)

lista_padre_las_casas = get_slices(padre_las_casas, amount)
lista_las_encinas = get_slices(las_encinas, amount)
lista_nielol = get_slices(nielol, amount)

lista_padre_las_casas = [i for i in lista_padre_las_casas if len(i) > length]
lista_las_encinas = [i for i in lista_las_encinas if len(i) > length]
lista_nielol = [i for i in lista_nielol if len(i) > length]

data = lista_padre_las_casas + lista_las_encinas + lista_nielol
X, y = [], []
for df in data:
    result = get_consecutive(df, length)
    X += result[0]
    y += result[1]

with open('data/data_product_03.pk', 'wb') as f:
    pickle.dump((X, y), f)
f.close()
