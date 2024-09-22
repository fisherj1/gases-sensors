import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import copy
import argparse

SENSORS_PATTERN = re.compile(r'R\d+_\d+')
DYNAMICS = [
    'linear',
    'linear_short',
    'steps_up',
    'steps_up_pulse',
    'steps_down',
    'steps_down_pulse'
]

def split_list(list, pattern):
    """
    Разделяет элементы списка на две группы в зависимости от соответствия регулярному выражению.

    Параметры:
    ----------
    lst : list
        Исходный список элементов, которые нужно разделить.
    pattern : re.Pattern
        Скомпилированное регулярное выражение, которое используется для разделения элементов.

    Возвращает:
    -------
    tuple
        Кортеж из двух списков:
        - first : list
            Список элементов, которые соответствуют регулярному выражению.
        - second : list
            Список элементов, которые не соответствуют регулярному выражению.
    """
    first = []
    second = []
    for name in list:
        if pattern.match(name):
            first.append(name)
        else:
            second.append(name)
    return first, second

def get_gas_name(list):
    """
    Извлекает имя газа из списка колонок, содержащих '_conc'.

    Параметры:
    ----------
    lst : list
        Список колонок, из которого нужно извлечь имя газа.

    Возвращает:
    -------
    str
        Имя газа, извлеченное из первой колонки, содержащей '_conc'.
    """
    return [name for name in list if '_conc' in name][0][:-5]

def melt_df(
        df, 
        gas_name, 
        sensors_columns, 
        not_sensors_columns, 
        sensors=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12']
    ):
    """
    Преобразует DataFrame в вертикальный формат, разделяет колонки датчиков и не датчиков,
    и выполняет дополнительные преобразования.

    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный DataFrame, который нужно преобразовать.
    gas_name : str
        Имя газа, которое нужно заменить в колонках.
    sensors_columns : list
        Список колонок, соответствующих показаниям датчиков.
    not_sensors_columns : list
        Список колонок, не соответствующих показаниям датчиков.
    sensors : list, optional
        Список имен датчиков в правильном порядке. По умолчанию ['R1', 'R2', ..., 'R12'].

    Возвращает:
    -------
    pandas.DataFrame
        Преобразованный DataFrame.
    """
    sensors_columns = copy.deepcopy(sensors_columns)
    not_sensors_columns = copy.deepcopy(not_sensors_columns)

    df_melted = df.melt(id_vars=not_sensors_columns, var_name='Sensor', value_name='Value')
    df_melted[['R', 'Tact']] = df_melted['Sensor'].str.split('_', expand=True)
    df_melted['Tact'] = df_melted['Tact'].astype(int)
    not_sensors_columns.append('Tact')
    df_melted = df_melted.pivot_table(index=not_sensors_columns, columns='R', values='Value').reset_index()
    df_melted.columns.name = None
    df_melted = df_melted.rename_axis(None, axis=1)
    df_melted = df_melted[not_sensors_columns+sensors]
    df_melted = df_melted.rename(columns={gas_name: 'gas', gas_name+'_conc': 'conc'})
    df_melted['gas'] = df_melted['gas'].apply(lambda x: gas_name if x == 1 else np.nan)
    
    return df_melted

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse():
    parser = argparse.ArgumentParser(description='Программа для обработки данных датчиков.')
    parser.add_argument('--path', type=str, default='/Users/cher/Documents/gases-sensors/data/exp1',  help='Путь к файлу с данными.')
    parser.add_argument('--sensors', type=str, nargs='+', default=['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12'],
                        help='Список датчиков. По умолчанию: ["R1", "R2", ..., "R12"].')
    parser.add_argument('--save-path', default='/Users/cher/Documents/gases-sensors/data/', type=str, help='Путь для сохранения обработанных данных.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    path = args.path
    dir_name = path.split('/')[-1]
    dir_name_new = dir_name + '_melted'

    save_path = os.path.join(args.save_path, dir_name_new)
    create_dir(save_path)

    for dynamic in DYNAMICS:
        print(f'Dynamic: {dynamic}')
        dynamic_path = os.path.join(save_path, dynamic)
        create_dir(dynamic_path)
        for fname in os.listdir(os.path.join(path, dynamic)):
            fname_full = os.path.join(path, dynamic, fname)
            df = pd.read_csv(fname_full)
            sensors_columns, not_sensors_columns = split_list(df.columns.to_list(), SENSORS_PATTERN)
            gas_name = get_gas_name(not_sensors_columns)
            df_melted = melt_df(df, gas_name, sensors_columns, not_sensors_columns)
            fname_save = fname[:-4] + f'_{gas_name}.csv'
            df_melted.to_csv(os.path.join(dynamic_path, fname_save), index=False)
