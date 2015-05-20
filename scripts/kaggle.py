import pandas as pd 
import numpy as np
import seaborn as sns
from pylab import savefig
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier

def get_response(df, responseColumn):
	return df[responseColumn]

def get_encoded_labels(labels):
	lbl_encoder = LabelEncoder()
	return lbl_encoder.fit_transform(labels)


otto_train = pd.read_csv('./train.csv/train.csv', index_col='id')
otto_train.target = get_encoded_labels(get_response(otto_train, 'target'))

sss = StratifiedShuffleSplit(otto_train.target.values, n_iter=3, train_size=1000, random_state=12)
train_idx, test_idx = next(iter(sss))

X_train = otto_train[otto_train.columns[:-1]].ix[train_idx]
y_train = otto_train.target.ix[train_idx]

forest = RandomForestClassifier(n_estimators=250, random_state=1728)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

five_most_important_features = [X_train.columns[indices[f]] for f in range(5)]
five_most_important_features.append('target')

df_to_plot = otto_train.ix[train_idx, five_most_important_features]

sns.pairplot(df_to_plot, hue='target', diag_kind='kde')
savefig('important_features.png')