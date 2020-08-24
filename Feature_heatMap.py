# Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


data = pd.read_csv('E:\M.Tech_Project\code\CSV\SUSY1.csv', encoding= 'unicode_escape')
X = data.iloc[:,0:20]  # independent columns
Y = data.iloc[:,0]  # target column i.e Signal


# Finding Correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index

# Plotting heat map
plt.figure(figsize=(25,25))
g = sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()
