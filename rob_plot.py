import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("cfiles/QuadX-Hover-v4_a2c_schedule0.csv")


max_val = (df.iloc[:, 7]).max()
print(max_val)

min_val = (df.iloc[:, 7]).min()
print(min_val)
