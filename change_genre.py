import pandas as pd
import glob
import numpy as np

df = pd.read_csv('./movie_genre_drama_fantasy.csv')

print(df['genre'].replace('Drama', 'drama'))
df['genre'].replace('Drama', 'drama', inplace=True)
df['genre'].replace('Fantasy', 'fantasy', inplace=True)
df['genre'].replace('Horror', 'horror', inplace=True)

print(df['genre'])
print(df['genre'].value_counts())

df.to_csv('./movie_genre_concat_dr_fa_ho.csv', index=False)