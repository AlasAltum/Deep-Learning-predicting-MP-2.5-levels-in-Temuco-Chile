import pandas as pd
from datetime import date, datetime

def get_season(date_time):
    """
    This function takes a datetime object and determines the season
    :param date_time: datetime object
    :return: season str
    """
    # dummy leap year to include leap days(year-02-29) in our range
    leap_year = 2000
    seasons = [('verano', (date(leap_year, 1, 1), date(leap_year, 3, 20))),
               ('otoño', (date(leap_year, 3, 21), date(leap_year, 6, 20))),
               ('invierno', (date(leap_year, 6, 21), date(leap_year, 9, 22))),
               ('primavera', (date(leap_year, 9, 23), date(leap_year, 12, 20))),
               ('verano', (date(leap_year, 12, 21), date(leap_year, 12, 31)))]

    if isinstance(date_time, datetime):
        date_time = date_time.date()
    # we don't really care about the actual year so replace it with our dummy leap_year
    date_time = date_time.replace(year=leap_year)
    # return season our date falls in.
    return next(season for season, (start, end) in seasons
                if start <= date_time <= end)

def create_season_column(data_set, date_column):
    """
    takes the get_season function and uses it to apply it in a df
    :param data_set: pandas df
    :param date_column: name of the column with the datetime objs
    :return: pandas df with the new column
    """
    # cloning the input dataset.
    local = data_set.copy()
    # The apply method calls a function on each row
    local['Estacion'] = local[date_column].apply(get_season)
    return local


def open_df(path, name):
    """
    Opens a generic csv of sinca
    :param path: path of the csv
    :param name: name to give to the column after the processing
    :return: returns a pandas df
    """
    datetime_format = {'FECHA (YYMMDD)': str, 'HORA (HHMM)': str}
    df = pd.read_csv(path, sep=';', dtype=datetime_format, decimal=',')
    df['datetime'] = pd.to_datetime(df['FECHA (YYMMDD)'] + ':' + df['HORA (HHMM)'], format='%y%m%d:%H%M')
    df.drop('Unnamed: 3', axis=1, inplace=True)
    df.rename({'Unnamed: 2': name}, axis=1, inplace=True)
    df.drop(['FECHA (YYMMDD)','HORA (HHMM)'], axis=1, inplace=True)
    return df


def multi_date(df):
    """
    takes a pd df and makes a multiindex and aplies the season
    :param df: df to apply on
    :return: df with the multi index and season
    """
    df = create_season_column(df, date_column='datetime')
    df.index = pd.MultiIndex.from_arrays([df.datetime.dt.year,
                                          df.datetime.dt.month,
                                          df.datetime.dt.day,
                                          df.datetime.dt.time],
                                          names=['Year', 'Month', 'Day','Time'])
    return df

def fillna_cols(df, cols, drop=False, col_name='merged data'):
    """
    this function combines many columns that are exclusive in the nans values
    :param df: pd df
    :param cols: cols to fillna
    :param drop: should it drop the original ones or not
    :param col_name: name of the new column
    :return: dataframe with the merged column
    """
    df = df.copy()
    df[col_name] = df[cols[0]]
    for i in cols[1:]:
        df[col_name].fillna(df[i], inplace=True)
    if drop:
        df.drop(cols, axis=1, inplace=True)
    return df

def process_mp25(path, col_name='merged data'):
    """
    this function is used only in the mp25 and is used to load it
    :param path: path to the df
    :param col_name: col name after merging column
    :return: dataframe of mp25
    """
    datetime_format = {'FECHA (YYMMDD)': str, 'HORA (HHMM)': str,
                       'Registros preliminares': float, 'Registros no validados': float}
    data = pd.read_csv(path, sep=';', dtype=datetime_format, decimal=',')
    data['datetime'] = pd.to_datetime(data['FECHA (YYMMDD)'] + ':' + data['HORA (HHMM)'], format='%y%m%d:%H%M')
    data.drop(['FECHA (YYMMDD)','HORA (HHMM)'], axis=1, inplace=True)
    data.drop('Unnamed: 5', axis=1, inplace=True)
    data = fillna_cols(data,
                       ['Registros validados',
                        'Registros preliminares',
                        'Registros no validados'],
                       drop=True,
                       col_name=col_name)
    return data