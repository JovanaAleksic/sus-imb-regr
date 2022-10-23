'''Main experiments conducted.
'''


import numpy as np
import pandas as pd
import os
import torch
import math
import time


from sampling import *
from learning import *
from metrics import *
from utils import *



def main(config, dataframe):

	# create instance of a class
	data = DataProcess(dataframe)
	
	# shuffle data in every run - important
	data.shuffle()

	# create balanced test dataset and separate train and test data
	data.split_data(config["test_size"])
	
	# Ribeiro 2011 relevance function
	# compute points of the target range to separate rare values from major 
	data.compute_phi(config["cutoff"])



	if config["sampling_method"] != 'SUSiter':

		# process dataset and obtain changed - sampled dataset
		sampler = SamplingFactory(data, config["ratio"], config["n_neighbors"], config["sampling_method"], config["blob_threshold"], config["spread_threshold"]).getSampler()
		X_sampled, y_sampled = sampler.sample()
		
		# train model on processed dataset
		y_pred = globals()[config["learning_method"]](X_sampled, y_sampled, data.X_test)

		# evaluate model
		evalue = globals()[config["metric"]](data.y_test, y_pred)

	else:

		# train model and process dataset
		model = NNiter(data, config["epochs"], config["n_neighbors"], config["blob_threshold"], config["spread_threshold"])

		# get predictions for X_test
		X_test = torch.FloatTensor(data.X_test)
		with torch.no_grad():
				y_pred = model.forward(X_test).view(-1).detach().numpy()
	
		# evaluate model
		evalue = globals()[config["metric"]](data.y_test, y_pred)






if __name__=="__main__":

	# repeat computation for statistical significance
	repeat_nb = 30

	for metric in ['RMSE']:
		for learning_method in ['NN']:   
			for sampling_method in ['SUSiter']:
				for dataset_type in ['standard']:

					# standard 15 datasets or synthetic highdimensional
					if dataset_type=='standard':
						# Location of datasets
						datasets_folder = "../Data/Standard/"  
						list_of_datasets = os.listdir(datasets_folder)
					elif dataset_type=='HD':
						datasets_folder = "../Data/HighDimensional/"  
						list_of_datasets = os.listdir(datasets_folder)
					else:
						datasets_folder = "../Data/MLP/"  
						list_of_datasets = os.listdir(datasets_folder)
						print(list_of_datasets)

				for k in [5, 7, 10]:
					for dataset_name in list_of_datasets:
						dataframe = pd.read_csv(datasets_folder + str(dataset_name)) 
						print(dataset_name)
						
						for i in range(repeat_nb):

								config = {
										"dataset_type": dataset_type,  
										"dataset_name": dataset_name,
										"sampling_method": sampling_method,
										"learning_method": learning_method,
										"metric": metric,

										# nn parameter
										"epochs": 300,
										"hidden": (20,5),

										# random under-sampling parameter
										"ratio": 0.5,
										
										# sus parameters
										"n_neighbors": k,
										"blob_threshold": 75,
										"spread_threshold": 0.5,

										"test_size": 0.2,
										# relevance cutoff
										"cutoff": 0.8  # default in SMOGN 0.5 - 0.8 used in their paper
										}


								main(config, dataframe)
									
					
								
									







