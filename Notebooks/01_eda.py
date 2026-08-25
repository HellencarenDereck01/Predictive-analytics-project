# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv("../data/student-mat.csv", sep=";")
# %%
df.head()
# %%
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
# %%
df.info()
# %%
df.describe()
