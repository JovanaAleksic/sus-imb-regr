'''
Sampling techniques for benchmarking SUS:
1. Original Dataset
2. Random undersampling - RU
3. SMOGN
4. Selective Under-sampling - SUS
5. SUSiter
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import random
import copy

# SMOGN
import smogn



##########################################################################################

class BaseSampler:
	def __init__(self, dataobject):
		self.X = dataobject.X_train
		self.y = dataobject.y_train
		self.array_of_indexes_major = dataobject.indexes_to_undersample

		total_list_y_index = range(len(self.y))
		self.array_of_indexes_minor = [index for index in total_list_y_index if index not in self.array_of_indexes_major]


	def sample(self):
		return self.X, self.y



##########################################################################################

class RU(BaseSampler):
	def __init__(self, dataobject, ratio):
		super().__init__(dataobject)
		self.ratio = ratio

	def sample(self):

		# pick indexes of less valuable samples to be included in the training
		random_indexes = np.random.choice(self.array_of_indexes_major, size=round(len(self.array_of_indexes_major)*self.ratio), replace=False).tolist()

		rus_index = self.array_of_indexes_minor + random_indexes		

		X_rus = self.X[rus_index]
		y_rus = self.y[rus_index]

		return X_rus, y_rus


##########################################################################################

class SMOGN(BaseSampler):
	def __init__(self, dataobject, ):
		super().__init__(dataobject)
		self.columns = dataobject.data.columns

	def sample(self):

		dataset_train = pd.DataFrame(data = np.concatenate((self.y.reshape(self.y.shape[0], 1), self.X), axis=1), columns = self.columns)

		dataset_smogn = smogn.smoter(data = dataset_train, y = self.columns[0], rel_thres = 0.8) 

		X_smogn = dataset_smogn.iloc[:, 1:].values
		y_smogn = dataset_smogn.iloc[:, 0].values

		return X_smogn, y_smogn


##########################################################################################

class SUS(BaseSampler):
	def __init__(self, dataobject, n_neighbors, blob_threshold, spread_threshold):
		super().__init__(dataobject)
		self.n_neighbors = n_neighbors
		self.blob_tr = blob_threshold
		self.spread_tr = spread_threshold


	def sample(self):

		# subset of data to be undersampled
		self.X_major = self.X[self.array_of_indexes_major]
		self.y_major = self.y[self.array_of_indexes_major]
		# indexes of rare data
		self.X_minor = self.X[self.array_of_indexes_minor]
		self.y_minor = self.y[self.array_of_indexes_minor]

		N = len(self.y_major) # data nb
		all_indexes = [*range(N)]
		grade_array = np.zeros(N)
		visited = np.zeros(N)
		avg_distances = np.zeros(N)

		undersampled_indexes = []

		# knn model
		knn = NearestNeighbors(n_neighbors=self.n_neighbors) # parameter to be set by user
		knn.fit(self.X_major) # only fit model to set of data points to be undersampled
		self.knn = knn

		for i in all_indexes:
			distances, neighbour_indexes = knn.kneighbors([self.X_major[i]], return_distance=True)
			distances = distances[0][1:]
			avg_distances[i] = np.mean(distances)

		# 75% blob_tr parameter to be set by the user
		self.blob = np.percentile(avg_distances, self.blob_tr)


		for i in all_indexes:

			if visited[i]!=1:

				distances, neighbour_indexes = knn.kneighbors([self.X_major[i]], return_distance=True)
				neighbour_indexes = neighbour_indexes[0][:] # removing itself with [1:]
				distances = distances[0][:]


				close_neighbour_boolean  = np.where(distances<self.blob, True, False) # whether neighbours are close or not
				close_neighbour_nb = close_neighbour_boolean.sum() # number of close neighbours


				if close_neighbour_nb==1:	 # itself, no close neighbours otherwise
					grade_array[i] = 2
					visited[i] = 1
				else:

					# main array indexes of close neighbours
					close_neighbour_indexes = [neighbour_indexes  for neighbour_indexes, close_neighbour_boolean in zip(neighbour_indexes, close_neighbour_boolean) if close_neighbour_boolean]

					# number of indexes to include 
					# close_neighbour_indexes changed in recursion - copy copy to avoid change
					# = operator changes the original array as well
					cluster_indxs = []  # initialize empty return list
					cluster_indxs = SUSReg.datadecision(copy.copy(close_neighbour_indexes), self.y_major, cluster_indxs, self.spread_tr)

					for index in cluster_indxs:
 						grade_array[index] = 1

					for close in close_neighbour_indexes:
						visited[close]=1

					visited[i]=1

		undersampled_indexes = [i for i in all_indexes if grade_array[i]==1 or grade_array[i]==2]

		X_sus = np.concatenate((self.X_minor, self.X_major[undersampled_indexes]), axis=0)
		y_sus = np.concatenate((self.y_minor, self.y_major[undersampled_indexes]), axis=0)

		self.grade_array = grade_array

		# just for experiments
		self.reduction = len(undersampled_indexes) / len(self.y_major)

		return X_sus, y_sus	


	@staticmethod
	def datadecision(close_neighbour_indexes, y_major, return_list, spread_tr):

		close_neighbour_y = np.zeros(len(close_neighbour_indexes))
		# loop through close neighbours
		for j in range(len(close_neighbour_indexes)): 
			close_neighbour_y[j] = y_major[close_neighbour_indexes[j]]

		average_y = np.mean(close_neighbour_y)
		variance_y = np.var(close_neighbour_y)

		if variance_y!=0:   
			y_spread = variance_y / average_y
		else: 
			y_spread = 0 


		# 0.5 parameter as well
		if y_spread < spread_tr:
			return_list.append(close_neighbour_indexes[np.argmin(abs(close_neighbour_y - average_y))]) # closest to average
			return return_list 
		else: 
			if len(close_neighbour_indexes) != 1:
				distant_y_position = np.argmax(abs(close_neighbour_y - average_y)) # remove the furthest from the average point
				return_list.append(close_neighbour_indexes[distant_y_position]) # add to the return list
				close_neighbour_indexes.pop(distant_y_position) # remove it for recurson
			else:
				return_list.append(close_neighbour_indexes[0])
				return return_list

		return SUSReg.datadecision(close_neighbour_indexes, y_major, return_list, spread_tr)  


##########################################################################################

class SUSiter(SUSReg):

	def __init__(self, dataobject, n_neighbors, blob_threshold, spread_threshold):
		super().__init__(dataobject, n_neighbors, blob_threshold, spread_threshold)

	def iter_sample(self):
		
		undersampled_indexes = []

		for i in range(len(self.y_major)): # total array size to be undersampled

			if self.grade_array[i]==2:
				undersampled_indexes.append(i)

			elif self.grade_array[i]==1:
				distances, neighbour_indexes = self.knn.kneighbors([self.X_major[i]], return_distance=True)
				neighbour_indexes = neighbour_indexes[0] # removing itself with [1:]
				distances = distances[0]
				close_neighbour_boolean  = np.where(distances<self.blob, True, False) # whether neighbours are close or not
				close_neighbour_indexes = [neighbour_indexes  for neighbour_indexes, close_neighbour_boolean in zip(neighbour_indexes, close_neighbour_boolean) if close_neighbour_boolean]
				
				undersampled_indexes.append(random.choice(close_neighbour_indexes))

		X_sus = np.concatenate((self.X_minor, self.X_major[undersampled_indexes]), axis=0)
		y_sus = np.concatenate((self.y_minor, self.y_major[undersampled_indexes]), axis=0)

		return X_sus, y_sus	


##########################################################################################

class SamplingFactory:
	def __init__(self, dataobject, ratio, n_neighbours, sample_method, blob_threshold, spread_threshold):
		self.method = sample_method
		self.data = dataobject
		self.ratio = ratio
		self.n_neighbors = n_neighbours
		self.blob_tr = blob_threshold
		self.spread_tr = spread_threshold


	def getSampler(self):
		if self.method == 'original':
			return BaseSampler(self.data)
		elif self.method == 'RU':
			return RU(self.data, self.ratio)
		elif self.method == 'SMOGN':
			return SMOGN(self.data)
		elif self.method == 'SUS':
			return SUS(self.data, self.n_neighbors, self.blob_tr, self.spread_tr)

