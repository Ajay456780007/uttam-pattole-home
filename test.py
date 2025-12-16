import pandas as pd
import numpy as np
from collections import Counter

m = pd.read_csv("Sub_Functions\classification_head.csv")

unique = m["classification_head"].values
u = Counter(unique)
print(u)
print(m.shape)
print(m.columns)


