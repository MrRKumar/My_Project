# Importing  Required Libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

data = pd.read_csv('E:\M.Tech_Project\Datasets\POKER HAND\poker-hand-training.csv', encoding= 'unicode_escape')
X = data.iloc[:,0:20]
y = data.iloc[:,-1]

# Applying SelectKBest class to extract top 10 best features of dataset
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=11)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# Concatinating two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(19,'Score'))  #print top 10 best features
