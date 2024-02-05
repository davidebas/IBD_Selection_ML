import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing
from scipy.stats import gaussian_kde
from tensorflow import keras
from plots import correlation_matrix_sig_bkg 

class DataPreprocessor:
	def __init__(self, input_dataset_sig, input_dataset_bkg, variables, variables_to_use):
		self.df_sig = pd.read_csv(input_dataset_sig, delimiter='	')
		self.df_bkg = pd.read_csv(input_dataset_bkg, delimiter='	')
		self.variables = variables
		self.variables_to_use = variables_to_use
		self.df_sig_histo = pd.DataFrame()
		self.df_bkg_histo = pd.DataFrame()

	def apply_transformations(self):
		self.apply_power_law()
		self.handle_missing_values()
		self.add_type_column()
		self.apply_fv_cut()
		self.resample_distributions()

		return self.df_sig_histo, self.df_bkg_histo

	def apply_power_law(self):
		self.df_sig["rp3"] = self.df_sig["rp"]**3.
		self.df_sig["rd3"] = self.df_sig["rd"]**3.
		self.df_bkg["rp3"] = self.df_bkg["rp"]**3.
		self.df_bkg["rd3"] = self.df_bkg["rd"]**3.

	def handle_missing_values(self):
		self.df_sig = self.df_sig.dropna()
		self.df_bkg = self.df_bkg.dropna()

	def add_type_column(self):
		type_bkg = np.array(1 for _ in range(len(self.df_bkg["ep"])))
		self.df_bkg["type"] = type_bkg

	def apply_fv_cut(self):
		R_Threshold_Max = 17.2 # Aggiungi il valore corretto
		self.df_bkg = self.df_bkg.loc[
			(self.df_bkg["rp3"] < R_Threshold_Max**3.) & (self.df_bkg["rd3"] <= R_Threshold_Max**3.)
		]

		type_sig = np.array(0. for _ in range(len(self.df_sig["ed"])))
		self.df_sig["type"] = type_sig

		if self.variables == 7:
			self.df_sig["QLpFlat"] = self.df_sig["QLp"] / self.df_sig["ep"]**PowerLawCoefficient_p
			self.df_bkg["QLpFlat"] = self.df_bkg["QLp"] / self.df_bkg["ep"]**PowerLawCoefficient_p
			self.df_sig["QLdFlat"] = self.df_sig["QLd"] / self.df_sig["ed"]**PowerLawCoefficient_d
			self.df_bkg["QLdFlat"] = self.df_bkg["QLd"] / self.df_bkg["ed"]**PowerLawCoefficient_d

	def resample_distributions(self):
		np.random.seed(10)
		
		kde_sig = gaussian_kde([self.df_sig[var] for var in self.variables_to_use], bw_method=0.01)
		kde_bkg = gaussian_kde([self.df_bkg[var] for var in self.variables_to_use], bw_method=0.01)

		resampled_points_bkg = kde_bkg.resample(len(self.df_sig))
		
		self.df_bkg_histo["rp3"] = resampled_points_bkg[0]
		self.df_bkg_histo["rd3"] = resampled_points_bkg[1]
		self.df_bkg_histo["ep"] = resampled_points_bkg[2]
		self.df_bkg_histo["ed"] = resampled_points_bkg[3]
		self.df_bkg_histo["deltar"] = resampled_points_bkg[4]
		
		if self.variables == 7:
			self.df_bkg_histo["QLpFlat"] = resampled_points_bkg[5]
			self.df_bkg_histo["QLdFlat"] = resampled_points_bkg[6]

		self.df_bkg_histo = self.df_bkg_histo.loc[(self.df_bkg_histo["rp3"] > 0.) & (self.df_bkg_histo["rd3"] > 0.)]
		type_bkg = np.array(1. for _ in range(len(self.df_bkg_histo["rp3"])))
		self.df_bkg_histo["truth"] = type_bkg
		self.df_bkg_histo = self.df_bkg_histo.reset_index(drop=True)

		self.df_sig_histo["rp3"] = self.df_sig["rp3"]
		self.df_sig_histo["rd3"] = self.df_sig["rd3"]
		self.df_sig_histo["ep"] = self.df_sig["ep"]
		self.df_sig_histo["ed"] = self.df_sig["ed"]
		self.df_sig_histo["deltar"] = self.df_sig["deltar"]
		
		if self.variables == 7:
			self.df_sig_histo["QLpFlat"] = self.df_sig["QLpFlat"]
			self.df_sig_histo["QLdFlat"] = self.df_sig["QLdFlat"]
			
class NeuralNetworkManager:
	def __init__(self, data_preprocessor):
		self.data_preprocessor = data_preprocessor
		
	#def resample_and_plot_correlation_matrices(self):
	#	self.resample_distributions()
	#self.plot_correlation_matrices()

	def prepare_train_test_validation(self):
		self.data_preprocessor.apply_transformations()

		type_sig = np.array(0. for i in range (0, len(self.data_preprocessor.df_sig_histo["rp3"])))
		self.data_preprocessor.df_sig_histo["truth"] = type_sig
		self.data_preprocessor.df_sig_histo = self.data_preprocessor.df_sig_histo.head(len(self.data_preprocessor.df_bkg_histo))
		
		# NN
		#train_sig, test_sig = train_test_split(self.data_preprocessor.df_sig_histo, test_size = 0.3, random_state=30)
		#test_sig, validation_sig = train_test_split(test_sig, test_size = 0.5, random_state=30)

		#train_bkg, test_bkg = train_test_split(self.data_preprocessor.df_bkg_histo, test_size = 0.3, random_state=30)
		#test_bkg, validation_bkg = train_test_split(test_bkg, test_size = 0.5, random_state=30)
		
		# BDT
		train_sig, test_sig = train_test_split(self.data_preprocessor.df_sig_histo, test_size = 0.3, shuffle = True)
		test_sig, validation_sig = train_test_split(test_sig, test_size = 0.5, shuffle = True)

		train_bkg, test_bkg = train_test_split(self.data_preprocessor.df_bkg_histo, test_size = 0.3, shuffle = True)
		test_bkg, validation_bkg = train_test_split(test_bkg, test_size = 0.5, shuffle = True)


		train = pd.concat([train_sig, train_bkg], axis = 0) #ignore_index = True)
		test = pd.concat([test_sig, test_bkg], axis = 0) #ignore_index = True)
		validation = pd.concat([validation_sig, validation_bkg], axis = 0)

		label_train = train["truth"].to_numpy()
		label_test_1 = test["truth"].to_numpy()
		label_validation_1 = validation["truth"].to_numpy()

		train = train.drop(labels = "truth", axis = 1)
		test = test.drop(labels = "truth", axis = 1)
		validation = validation.drop(labels = "truth", axis = 1)

		# convert class vectors to binary class matrices, e.g. for use with categorical_crossentropy
		label_train = keras.utils.to_categorical(label_train, 2) #2 is the number of the classes
		label_test = keras.utils.to_categorical(label_test_1, 2)
		label_validation = keras.utils.to_categorical(label_validation_1, 2)

		return train, test, test_sig, test_bkg, validation, label_train, label_test, label_test_1, label_validation

