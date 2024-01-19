#### Selection cuts for IBD analysis with Neural Network methods
#### Davide Basilico (davide.basilico@mi.infn.it), Gloria Senatore (gloria.senatore@mi.infn.it) - 2023 December 02

def main():

	#### Import libraries
	from lib import import_lib
	from lib import read_settings

	time, np, os, plt, pd, LogNorm, gaussian_kde, sys, sns, ListedColormap, argparse, configparser, roc_curve, train_test_split, auc, tf, keras, SGD, Adam, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, ReduceLROnPlateau, Sequential, Dense, Activation, Dropout, K, get_custom_objects = import_lib()

	#### Parsing the inputs and defining the analysis configurations
	
	start_time = time.time()
	parser = argparse.ArgumentParser(description='Parsing')
	parser.add_argument('--config', help='Configuration file')

	args = parser.parse_args()
	
	if args.config:
		impostazioni = read_settings(args.config)

		name = impostazioni.get('name', 'Default')
		epochs = int(impostazioni.get('epochs', 10))
		training = int(impostazioni.get('training', 10))
		variables = int(impostazioni.get('variables', 5))
		InputDataset_Sig = impostazioni.get('InputDataset_Sig', 'InputDataset_Sig.txt')
		InputDataset_Bkg = impostazioni.get('InputDataset_Bkg', 'InputDataset_Bkg.txt')
		
		R_Threshold_Max = impostazioni.get('R_Threshold_Max',17.2)
		DeltaR_Max = impostazioni.get('DeltaR_Max',1.5)
		Ep_Threshold_Min = impostazioni.get('Ep_Threshold_Min',0.7)
		Ep_Threshold_Max = impostazioni.get('Ep_Threshold_Max',12)		
		Ed_Threshold_Min_1 = impostazioni.get('Ed_Threshold_Min_1',1.9)
		Ed_Threshold_Max_1 = impostazioni.get('Ed_Threshold_Max_1',2.5)		
		Ed_Threshold_Min_2 = impostazioni.get('Ed_Threshold_Min_2',4.9)
		Ed_Threshold_Max_2 = impostazioni.get('Ed_Threshold_Max_2',5.5)		
		QLp_Threshold_Min = impostazioni.get('Qlp_Threshold_Min',15900)
		QLd_Threshold_Max = impostazioni.get('Qlp_Threshold_Max',18000)
		PowerLawCoefficient_p = impostazioni.get('PowerLawCoefficient_p',0.4977)
		PowerLawCoefficient_d = impostazioni.get('PowerLawCoefficient_d',0.4929)
		batch_size = int(impostazioni.get('batch_size', 212))

		print("-- INPUTS")
		print(f"Name: {name}")
		print(f"Epochs: {epochs} ")
		print(f"Training: {training} ")
		print(f"Variables: {variables} ")
		print(f"Input dataset signal: {InputDataset_Sig}")
		print(f"Input dataset background: {InputDataset_Bkg}")
		print(f"Batch size: {batch_size} ")
		
		from lib import create_folder_output
		create_folder_output(name)
		path = name + '/' + name
			
	#### Building the signal and background datasets and loose cuts
	
	prompt_cut = [R_Threshold_Max, Ep_Threshold_Min, Ep_Threshold_Max, DeltaR_Max]
	delayed_cut = [R_Threshold_Max, Ed_Threshold_Min_1, Ed_Threshold_Max_1, Ed_Threshold_Min_2, Ed_Threshold_Max_2]
	eff_meta = []
	pur_meta = []

	df_sig = pd.read_csv(InputDataset_Sig, delimiter='	')
	df_sig["rp3"] = df_sig["rp"]**3.
	df_sig["rd3"] = df_sig["rd"]**3.
	df_sig = df_sig.dropna()
	
	df_bkg = pd.read_csv(InputDataset_Bkg, delimiter='	')
	df_bkg["rp3"] = df_bkg["rp"]**3.
	df_bkg["rd3"] = df_bkg["rd"]**3.
	df_bkg = df_bkg.dropna()
	
	type_bkg = np.array(1 for i in range (0, len(df_bkg["rp3"]))) #type 1 is background
	df_bkg["type"] = type_bkg
	
	df_bkg = df_bkg.loc[(df_bkg["rp3"] < R_Threshold_Max**3.) & (df_bkg["rd3"] <= R_Threshold_Max**3.)] # FV cut
	
	type_sig = np.array(0. for i in range (0, len(df_sig["rp3"]))) #type 0 is signal
	df_sig["type"] = type_sig
	
	if(variables == 7 ):
		df_sig["QLpFlat"] = df_sig["QLp"]/df_sig["ep"]**PowerLawCoefficient_p
		df_bkg["QLpFlat"] = df_bkg["QLp"]/df_bkg["ep"]**PowerLawCoefficient_p
		df_sig["QLdFlat"] = df_sig["QLd"]/df_sig["ed"]**PowerLawCoefficient_d
		df_bkg["QLdFlat"] = df_bkg["QLd"]/df_bkg["ed"]**PowerLawCoefficient_d
	
	#### Resampling of distributions with gaussian KDEs

	np.random.seed(10)
	
	print("NUMBER OF EVENTS FOR EACH DATASET DF_SIG AND DF_BKG")
	print(len(df_sig))
	print(len(df_bkg))

	variables_to_use = ["rp3", "rd3", "ep", "ed", "deltar"]
	projection_values_variables_to_use = [1000, 1000, 5, 2.2, 1.5]
	minimum_variables = [0, 0, Ep_Threshold_Min, Ed_Threshold_Min_1, 0]
	maximum_variables = [R_Threshold_Max**3, R_Threshold_Max**3, Ep_Threshold_Max, Ed_Threshold_Max_1, 4]

	if variables == 7:
	    variables_to_use.extend(["QLpFlat", "QLdFlat"])
	    projection_values_variables_to_use.extend([20000,20000])
	    minimum_variables.extend([0,0])
	    minimum_variables.extend([60000,60000])
	    
	kde_sig = gaussian_kde([df_sig[var] for var in variables_to_use], bw_method=0.01)
	kde_bkg = gaussian_kde([df_bkg[var] for var in variables_to_use], bw_method=0.01)
	
	resampled_points_bkg = kde_bkg.resample(len(df_sig)) #96527)
	
	df_bkg_histo = pd.DataFrame()						# defining the 5D or 7D bkg dataframe for the re-sampled variables 
	df_bkg_histo["rp3"] = resampled_points_bkg[0]
	df_bkg_histo["rd3"] = resampled_points_bkg[1]
	df_bkg_histo["ep"] = resampled_points_bkg[2]
	df_bkg_histo["ed"] = resampled_points_bkg[3]
	df_bkg_histo["deltar"] = resampled_points_bkg[4]
	
	if variables == 7 :
		df_bkg_histo["QLpFlat"] = resampled_points_bkg[5]
		df_bkg_histo["QLdFlat"] = resampled_points_bkg[6]
	
	df_bkg_histo = df_bkg_histo.loc[(df_bkg_histo["rp3"] > 0.) & (df_bkg_histo["rd3"] > 0.)]	# discard possible non-physical events with r<0, which could be created by the re-sampling
	type_bkg = np.array(1. for i in range (0, len(df_bkg_histo["rp3"])))
	df_bkg_histo["truth"] = type_bkg
	
	
	df_sig_histo = pd.DataFrame()						# defining the 5D or 7D sig dataframe for the re-sampled variables
	df_sig_histo["rp3"] = df_sig["rp3"] #resampled_points_sig[0]
	df_sig_histo["rd3"] = df_sig["rd3"] #resampled_points_sig[1]
	df_sig_histo["ep"] = df_sig["ep"] #resampled_points_sig[2]
	df_sig_histo["ed"] = df_sig["ed"] #resampled_points_sig[3]
	df_sig_histo["deltar"] = df_sig["deltar"] #resampled_points_sig[4]
	df_sig_histo = df_sig_histo.head(len(df_bkg_histo))
	
	if(variables==7):
		df_sig_histo["QLpFlat"] = df_sig["QLpFlat"]
		df_sig_histo["QLdFlat"] = df_sig["QLdFlat"]

	#### Plotting Correlation Matrices of resampled signal and background datasets
	from plots import correlation_matrix_sig_bkg
	correlation_matrix_sig_bkg(df_sig_histo,df_bkg_histo,path) 	

	#### Definition of train, test and validation datasets, and of the architecture of the neural network.
	type_sig = np.array(0. for i in range (0, len(df_sig_histo["rp3"])))
	df_sig_histo["truth"] = type_sig
	df_sig_histo = df_sig_histo.head(len(df_bkg_histo))
	
	train_sig, test_sig = train_test_split(df_sig_histo, test_size = 0.3, random_state=30)
	test_sig, validation_sig = train_test_split(test_sig, test_size = 0.5, random_state=30)
	
	train_bkg, test_bkg = train_test_split(df_bkg_histo, test_size = 0.3, random_state=30)
	test_bkg, validation_bkg = train_test_split(test_bkg, test_size = 0.5, random_state=30)
	
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
		
	#### Training of neural network

	from TrainingManager import TrainingManager_NN
	manager = TrainingManager_NN(variables, training, epochs, name)
	manager.train_model(train, label_train, validation, label_validation, test, label_test)	
	
	label = manager.label	
	score = manager.score
	score_test = manager.score_test
	hist_acc = manager.hist_acc
	hist_val_acc = manager.hist_val_acc
	hist_loss = manager.hist_loss
	hist_val_loss = manager.hist_val_loss
	model = manager.model
	mean_acc = manager.mean_acc
	mean_val_acc = manager.mean_val_acc
	mean_loss = manager.mean_loss
	mean_val_loss = manager.mean_val_loss

	# Plot history for loss and accuracy

	from plots import plot_mean_model_history
	plot_mean_model_history(mean_acc, mean_val_acc, mean_loss, mean_val_loss, path)
		
	label_test_1 = 1. - label_test_1  # Now: 1 is signal, 0 is background

	predict = np.zeros((len(label[0]), training))  # label_pred_keras))
	for j in range(training):
		for i in range(len(label[0])):
        		predict[i][j] = label[j][i][0] if i < len(test_sig) else 1. - label[j][i][1]
	
	label_test_1 = label_test_1.astype(int)
	
	fpr_list, tpr_list, thresholds_list, meta_list, auc_list = [], [], [], [], []
	
	for i in range (0, training):
	  fpr_keras, tpr_keras, thresholds_keras = roc_curve(label_test_1, predict.T[i], pos_label = 1) #fpr = false positive rate. tpr = true positive rate
	  fpr_list.append(fpr_keras)
	  tpr_list.append(tpr_keras)
	  thresholds_list.append(thresholds_keras)
	  meta = np.where(thresholds_list[i] < 0.51)[0][0]
	  meta_list.append(meta)
	  auc_keras = auc(fpr_list[i], tpr_list[i])
	  auc_list.append(auc_keras)
	
	
	for i in range (0, training):
	  eff_meta.append(tpr_list[i][meta_list[i]])
	  pur_meta.append(1. - fpr_list[i][meta_list[i]])
	
	print("Training with the highest efficiency: {}. Efficiency = {:.4f}".format(np.argmax(eff_meta), np.max(eff_meta)))
	print("Training with the highest purity: {}. Purity = {:.4f}".format(np.argmax(pur_meta), np.max(pur_meta)))
	print("Training with the highest AUC: {}. AUC = {:.4f}".format(np.argmax(auc_list), np.max(auc_list)))

	auc_list = np.array(auc_list)
		
	mean_eff = sum(eff_meta)/training
	mean_pur = sum(pur_meta)/training
	mean_auc = sum(auc_list)/training
	
	dev_eff = np.sqrt(np.var(eff_meta, ddof = 1)/(training)) # standard error for the mean
	dev_pur = np.sqrt(np.var(pur_meta, ddof = 1)/(training)) # standard error for the mean
	dev_auc = np.sqrt(np.var(auc_list, ddof = 1)/(training))

	#IBD cuts efficiency/purity calculation
	from IBD_Std_Cuts import IBD_Std_Cuts
	tpr_cuts, fpr_cuts = IBD_Std_Cuts(test_sig, test_bkg, prompt_cut, delayed_cut)
		
	"""##### Continuo con l'analisi : Grafico della ROC curve"""
	
	#Cella per tracciare la curva media
	ar = np.linspace (0., 1, 2000)
	purity = np.zeros(shape = (len(ar),))
	
	for j in range (0, len(ar)):
		for i in range (0, training):
			sel = np.where(tpr_list[i] >= ar[j])[0][0]
			purity[j] = purity[j] + 1. - fpr_list[i][sel]
		purity[j] = float(purity[j])/training
	
	#Trascrivo su file la purity della curva media
	with open(path + '_Purity_median_curve.txt', 'w') as writefile:
		for j in range (0, len(ar)):
			writefile.write(str(purity[j]) + "\n")
		writefile.write("Average efficiency: {:.4f} +- {:.4f}".format(mean_eff, dev_eff))
		writefile.write("Average purity: {:.4f} +- {:.4f}".format(mean_pur, dev_pur))
		writefile.write("Average AUC: {:.4f} +- {:.4f}".format(mean_auc, dev_auc))
	
	#### Plot ROC curve - it illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. The critical point here is “binary classifier” and “varying threshold”. AUC is one single number to summarize a classifier’s performance by assessing the ranking regarding separation of the two classes. The higher, the better.
	
	from plots import plot_roc_curve
	plot_roc_curve(training, path, tpr_list, fpr_list, mean_auc, dev_auc, mean_eff, dev_eff, mean_pur, dev_pur, ar, purity, tpr_cuts, fpr_cuts)
	
	#### Histograms of the decision of NN

	num_best_model = np.argmax(pur_meta) # Load the model with the best purity
	print("Best model: number " + str(num_best_model) + ", purity " +  str(pur_meta[num_best_model]))
	
	pred_col = []
	for i in range (0, len(test)):
	    if(label[num_best_model][i][0] >= 0.5): pred_col.append("Signal")
	    else: pred_col.append("Background")
	        
	type_bkg = np.array(1. for i in range (0, len(test_bkg["rp3"])))
	test_bkg["truth"] = type_bkg
	
	type_sig = np.array(0. for i in range (0, len(test_sig["rp3"])))
	test_sig["truth"] = type_sig
	
	histo_data = pd.concat([test_sig, test_bkg], axis = 0)
	histo_data["prediction"] = pred_col
	
	##### Plotting Outputs
	
	# Plotting 1D distribution of variables
	from plots import plots_variables_1D_distributions
	plots_variables_1D_distributions(histo_data,variables,path)
	
	#### Neural network decision boundaries projected on a 2-D space
	
	name_x = "rp3"
	name_y = "ep"
	
#	model.load_weights(name + "/model_" + name + ".h." + str(num_best_model))
	model.load_weights(name + "/model_" + name + ".h." + str(0))

	from decision_boundaries_plotter import DecisionBoundariesPlotter
	from itertools import combinations
	
	for i in range (0,training):
		blank = path
		model.load_weights(name + "/model_" + name + ".h." + str(i))
		path = path + str(i) + ".pdf"
		decision_boundaries_plotter = DecisionBoundariesPlotter(model, "ep", "deltar", variables_to_use, projection_values_variables_to_use, minimum_variables, maximum_variables, variables, path)
		decision_boundaries_plotter.run()
		path = blank
	
	
	for arg1, arg2 in combinations(variables_to_use, 2):
		decision_boundaries_plotter = DecisionBoundariesPlotter(model, arg1, arg2, variables_to_use, projection_values_variables_to_use, minimum_variables, maximum_variables, variables, path)
		decision_boundaries_plotter.run()
		
	print("##### END #####")
	end_time = time.time()
	duration_in_seconds = end_time - start_time
	hours, remainder = divmod(duration_in_seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	print(f"Duration time: {int(hours)} h {int(minutes)} min {int(seconds)} s.")	

if __name__ == '__main__':
    main()
