import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ecommerce.csv')
 
head = df.head()

print(head.to_string())

info = df.info()

describe = df.describe()