import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

def load_csv(filename, index_col):
	return pd.read_csv(filename, index_col=index_col)

def get_columns(df):
	return df.columns

def get_features(df, featureList):
	return df[featureList]

def get_response(df, responseColumn):
	return df[responseColumn]

def get_encoded_labels(labels):
	lbl_encoder = LabelEncoder()
	return lbl_encoder.fit_transform(labels)

def compare(row1, row2, i, j, threshold):
	ncorri = len(np.where( row1 > threshold )[0])
	ncorrj = len(np.where( row2 > threshold )[0])

	return (ncorri, i) if ( ncorri > ncorrj ) else ( ncorrj, j)

def max_corr_variable(corr_matrix, threshold=0.7):
	best = 1
	nrows = corr_matrix.shape[0]

	for i in range(nrows - 1):
		for j in range(i+1, nrows):
			highest_btw_pair, idx = compare(corr_matrix.ix[i].values, corr_matrix.ix[j].values, i, j, threshold=threshold)

			if highest_btw_pair > best:
				best = highest_btw_pair
				max_corr_variable = corr_matrix.index[idx]

	return max_corr_variable

def get_stratified_shuffle_splits(labels, n_iter=3, train_size=1000, random_state=144):
	sss = StratifiedShuffleSplit(labels, n_iter=n_iter, train_size=train_size, random_state=random_state)
	train_idx, test_idx = next(iter(sss))

	return (train_idx, test_idx)

def get_class_distribution(df, totalSum):
	return dict([(idx, df.ix[idx] / (totalSum * 1.)) for idx in df.index])


def get_smaller_dataset(X, y, train_size):
	train_idx, test_idx = get_stratified_shuffle_splits(y, 3, train_size=train_size)
	return (X.ix[train_idx], y.ix[train_idx])


def make_submission(y_pred, filename):
	submission = pd.read_csv('./sampleSubmission.csv')
	submission.set_index('id', inplace=True)
	submission[:] = y_pred
	submission.to_csv('./submissions/' + filename);