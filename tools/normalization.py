def norm_max(df, sensors):
    df_copy = df.copy()
    df_copy[sensors] = df_copy[sensors]/df_copy[sensors].max()
    return df_copy

def norm_max_tact(df, sensors):
    df_copy = df.copy()
    # Нормализация значений по максимуму по Tact
    for sensor in sensors:
        max_values = df_copy.groupby('Tact')[sensor].transform('max')
        df_copy[sensor] = df_copy[sensor] / max_values
    return df_copy
