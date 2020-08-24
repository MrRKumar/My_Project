# Importing Required Libraries
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('E:\M.Tech_Project\Datasets\POKER HAND\poker-hand-training.csv', encoding= 'unicode_escape')
X = data.iloc[:,0:20]  # independent columns
y = data.iloc[:,-1]    # target column i.e Signal

# Training Model
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)

# Plotting graph of feature importance for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 12))
feat_importances.nlargest(11).plot(kind='barh')
plt.show()