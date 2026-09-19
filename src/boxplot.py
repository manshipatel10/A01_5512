from sklearn.datasets import fetch_california_housing
import pandas as pd

#Load Califirnia housing dataset
housing=fetch_california_housing(as_frame=True)

#Features + Target as a single DataFrame
df = housing.frame 

#Quick check
print(df.head())
print(df.shape)

#save the boxplot
import matplotlib.pyplot as plt

df.boxplot()
plt.savefig("california_housing_boxplot.png")
plt.show()