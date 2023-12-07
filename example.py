import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = r"臺南市112年上半年道路交通事故原因傷亡統計.csv"
df_filtered = df[(df['發生時間'] > 0) & (df['發生時間'] < 120000)]
df_filtered.to_csv("test.csv", index=False)