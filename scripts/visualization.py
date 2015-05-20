import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pylab import savefig

def visualize_clusters(X, y):
	df = pd.DataFrame({'x': X[:, 0], 'y': X[:, 1], 'label': y})

	colors = ['r', 'b', 'g', 'y', 'darkgreen', 'darkblue', 'cyan', 'black', 'brown']

	plt.figure(figsize=(10, 10))

	for label, color in zip(df['label'].unique(), colors):
	 	mask = df['label'] == label
	 	plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)

	plt.legend(loc='best')

def plot_confusion(cm, target_names, title='Confusion matrix'):
	
	plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title(title)
	plt.colorbar()

	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names, rotation=60)
	plt.yticks(tick_marks, target_names)
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')

	plt.tight_layout()

def visualize_important_variables(df_to_plot):
	sns.pairplot(df_to_plot, hue='target', diag_kind='kde')
	savefig('D:/Kaggle/Submission/Otto-Group/scripts/visualizations/important_features.png')
