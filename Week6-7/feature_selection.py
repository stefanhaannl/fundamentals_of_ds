import numpy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
plt.style.use('ggplot')
import pprint as pp
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import random

    
def correlation_matrix(df):
	"""
	Input: dataframe
	Output: correlation matrix
	"""
	return df.corr(method='pearson')

def plot_correlation(corr_matrix):
	"""
	Input: correlation matrix
	Output: plots of the corr matrix
	"""
	pp.pprint(corr_matrix)
	headers = list(df.columns.values)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(corr_matrix)
	fig.colorbar(cax)
	ax.set_xticklabels(['']+headers)
	ax.set_yticklabels(['']+headers)

	plt.show()
	
def normalize_df(df):
	"""
	Input: correlation matrix
	Output: plots of the corr matrix
	"""
	x = df.values
	headers = list(df.columns.values)
	x_scaled = preprocessing.scale(x)
	df = pd.DataFrame(x_scaled, columns = headers)
	return df
	
	
def variance_columns(df_norm):
	"""
	Input: normalized dataframe
	Output: dataframe without zero variance column
	"""
	headers = list(df_norm.columns.values)
	selector = VarianceThreshold()
	x = df_norm.values
	x = selector.fit_transform(df_norm)
	indexlist = selector.get_support(indices = True)
	headers1 = []
	for i in range(0, len(indexlist)):
		indexs = indexlist[i]
		term = headers[indexs]
		headers1.append(term)
	df = pd.DataFrame(x, columns = headers1)
	return df
	
if __name__ == "__main__": 
	
	# read dataframe
	df = pd.read_pickle('data/likes_comments_features.pkl')
	df = df.iloc[:5]
	
	# delete non numeric columns
	del df['image_id']
	del df['user_id']
	
	pp.pprint(df)
	# correlation matrix
	corr_matrix = correlation_matrix(df)
	
	# plot correlation matrix
	#plot_correlation(corr_matrix)
		
	# normalize for variance thresholding
	df_norm = normalize_df(df)
	pp.pprint(df_norm)

	# variance threshold input has to be normalized
	df_var = variance_columns(df_norm)
	pp.pprint(df_var)
