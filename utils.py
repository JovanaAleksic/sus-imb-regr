import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from phi_ctrl_points import *
from phi import *





class DataProcess:
	def __init__(self, data):
		self.data = data

		# extract categorical columns, change datatype from object to categorical
		cat_columns = data.select_dtypes(['object']).columns
		self.data[cat_columns] = data[cat_columns].astype('category')
		self.data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

		self.X = self.data.iloc[:, 1:].values  # feature values
		self.y = self.data.iloc[:, 0].values	# target values


		self.X_train = self.X  # feature values
		self.y_train = self.y	# target values
		self.sorted_y_train = np.sort(self.y_train)



	# def shuffle(self):
	# 	self.data = self.data.sample(frac=1).reset_index(drop=True) 
	# 	# extract categorical columns, change datatype from object to categorical
	# 	cat_columns = self.data.select_dtypes(['object']).columns
	# 	self.data[cat_columns] = self.data[cat_columns].astype('category')
	# 	self.data[cat_columns] = self.data[cat_columns].apply(lambda x: x.cat.codes)

	# 	self.X = self.data.iloc[:, 1:].values  # feature values
	# 	self.y = self.data.iloc[:, 0].values	# target values

	# 	self.X_train = self.X  # feature values
	# 	self.y_train = self.y	# target values
	
	# # create balanced test data
	# def split_data(self, test_size):

	# 	# choose balanced test dataset
	# 	ymin = min(self.y)
	# 	ymax = max(self.y)

	# 	size_of_test_dataset = round(len(self.y)*test_size) # number of test datapoints
	# 	random_points = np.random.uniform(low=ymin, high=ymax, size=size_of_test_dataset).tolist() # spread uniform random points in target range

	# 	test_indexes = []
	# 	for point in random_points:
	# 		# closest y values in a dataset to the random point
	# 		closestValue = min(self.y, key=lambda x:abs(x-point))
	# 		# index of that closest value point
	# 		index_closestValue = np.where(self.y==closestValue)[0][0]
	# 		test_indexes.append(index_closestValue)


	# 	all_indexes = [*range(len(self.y))]
	# 	train_indexes = list(set(all_indexes) - set(test_indexes)) # points in all indexes but not in test 

	# 	self.X_train = self.X[train_indexes]
	# 	self.y_train = self.y[train_indexes]
	# 	self.X_test = self.X[test_indexes]
	# 	self.y_test = self.y[test_indexes]

	# 	# normalize data
	# 	scaler = StandardScaler()
	# 	self.X_train = scaler.fit_transform(self.X_train)
	# 	self.X_test = scaler.transform(self.X_test)

	# 	self.sorted_y_train = np.sort(self.y_train)



	def compute_phi(self, cutoff):

		# implementation of relevance function
		# phi_ctrl_pts, phi and box_plot_stats implemented by https://github.com/nickkunz/smogn/ 
		phi_params = phi_ctrl_pts(y = self.y_train)
		y_phi = phi(y = self.y_train, ctrl_pts = phi_params) # relevance function finished
		y_phi = np.asarray(y_phi)
		self.phi_zip = zip(y_phi, self.y_train)

		# computation of control points
		all_indexes = [*range(len(self.y_train))]
		index_boolean = np.where(y_phi<cutoff, True, False)
		indexes_to_undersample = [all_indexes  for all_indexes, index_boolean in zip(all_indexes, index_boolean) if index_boolean]

		self.indexes_to_undersample = indexes_to_undersample









