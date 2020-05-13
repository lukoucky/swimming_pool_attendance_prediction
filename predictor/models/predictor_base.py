from sklearn.metrics import mean_squared_error, make_scorer
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import random
import pickle
import os

class PredictorBase(ABC):
	def __init__(self):
		"""
		Constructor
		"""
		pass

	@abstractmethod
	def fit(self, X, y, silent=True):
		"""
		Fit predictor on training data.
		:param X: List of numpy arrays with traininng data shape (n_samples, n_features)
		:param y: List of target values (attendance) with length n_samples
		:param silent: Boolean flag. If True method does not prints any info about
					   fitting progress. True is default value
		"""
		pass

	@abstractmethod
	def predict(self, X):
		"""
		Predicts attendance from given feature vector X
		:param X: List of numpy arrays with data shape (n_samples, n_features)
		:return: List with attendance, length n_samples
		"""
		pass

	@abstractmethod
	def load_predictor(self, pickle_path=None):
		"""
		Loads fitted predictor from pickle_path. If path is not provided
		default path based on predictor name is used.
		:param pickle_path: String with path to saved pickle
		"""
		pass

	@abstractmethod
	def save_predictor(self, pickle_path=None):
		"""
		Save fitted predictor to pickle_path. If path is not provided
		default path based on predictor name is used.
		:param pickle_path: String with path where to save pickle
		"""
		pass
