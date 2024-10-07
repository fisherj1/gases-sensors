def norm_max(df, sensors):
    df_copy = df.copy()
    df_copy[sensors] = df_copy[sensors]/df_copy[sensors].max()
    return df_copy
