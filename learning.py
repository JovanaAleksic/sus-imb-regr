'''
Learning methods:
NN --> Neural Networks
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from sampling import *


##########################################################################################

def NN(X_train, y_train,  X_test):

	# define NN 
	class Model(nn.Module):

		def __init__(self, in_fetures=X_train.shape[1], h1=8, h2=4, h3=2, out_features=1):
			super().__init__()
			self.fc1 = nn.Linear(in_fetures, h1)
			self.fc2 = nn.Linear(h1, h2)
			self.fc3 = nn.Linear(h2, h3)
			self.out = nn.Linear(h3, out_features)


		def forward(self, x):
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = self.out(x)

			return x


	model = Model()

	X_train = torch.FloatTensor(X_train)
	X_test = torch.FloatTensor(X_test)
	y_train = torch.FloatTensor(y_train) 

	criterion = nn.MSELoss() # loss function
	optimizer = torch.optim.Adam(model.parameters(), 0.01)

	epochs = 300
	losses = []

	for i in range(epochs):
		# X_train, y_train --> subsample from the dataset in a special way
		y_pred = model.forward(X_train)
		y_pred = y_pred.view(-1)  
		loss = torch.sqrt(criterion(y_pred, y_train))

		losses.append(loss.item())

		# if i%10==0:
		# print(f'epoch {i} and loss is: {loss}')

		# backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	final_y_pred = model.forward(X_test).view(-1).detach().numpy()

	return final_y_pred

##########################################################################################

def NNiter(data, epochs, n_neighbors, blob_tr, spread_tr):

	susiter =  SUSiter(data, n_neighbors, blob_tr, spread_tr)
	X_sus, y_sus = susiter.sample()  # sus

	# define NN manually 
	class Model(nn.Module):

		def __init__(self, in_fetures=X_sus.shape[1], h1=8, h2=4, h3=2, out_features=1):
			super().__init__()
			self.fc1 = nn.Linear(in_fetures, h1)
			self.fc2 = nn.Linear(h1, h2)
			self.fc3 = nn.Linear(h2, h3)
			self.out = nn.Linear(h3, out_features)


		def forward(self, x):
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = self.out(x)

			return x


	model = Model()

	criterion = nn.MSELoss() # loss function
	optimizer = torch.optim.Adam(model.parameters(), 0.01)

	losses = []

	for i in range(epochs):

		X_train = torch.FloatTensor(X_sus)
		y_train = torch.FloatTensor(y_sus) 

		y_pred = model.forward(X_train)
		y_pred = y_pred.view(-1)  
		loss =  torch.sqrt(criterion(y_pred, y_train))

		losses.append(loss.item())

		# backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		X_sus, y_sus = susiter.iter_sample() # sample different data

	return model

