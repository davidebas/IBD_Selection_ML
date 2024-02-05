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
		method = impostazioni.get('method', 'BDT')
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
		All2DPlots = impostazioni.get('All2DPlots', 'yes')

		print("-- INPUTS")
		print(f"Name: {name}")
		print(f"Method: {method} ")
		print(f"Epochs: {epochs} ")
		print(f"Training: {training} ")
		print(f"Variables: {variables} ")
		print(f"Input dataset signal: {InputDataset_Sig}")
		print(f"Input dataset background: {InputDataset_Bkg}")
		print(f"Batch size: {batch_size} ")
		print(f"2D plots: all? : {All2DPlots} ")
				
		from lib import create_folder_output
		create_folder_output(name)
		path = name + '/' + name
			
	#### Building the signal and background datasets and loose cuts	
	prompt_cut = [R_Threshold_Max, Ep_Threshold_Min, Ep_Threshold_Max, DeltaR_Max]
	delayed_cut = [R_Threshold_Max, Ed_Threshold_Min_1, Ed_Threshold_Max_1, Ed_Threshold_Min_2, Ed_Threshold_Max_2]
	eff_meta = []
	pur_meta = []

	variables_to_use = ["rp3", "rd3", "ep", "ed", "deltar"]
	projection_values_variables_to_use = [1000, 1000, 5, 2.2, 1.5]
	minimum_variables = [0, 0, Ep_Threshold_Min, Ed_Threshold_Min_1, 0]
	maximum_variables = [R_Threshold_Max**3, R_Threshold_Max**3, Ep_Threshold_Max, Ed_Threshold_Max_1, 4]

	if variables == 7:
	    variables_to_use.extend(["QLpFlat", "QLdFlat"])
	    projection_values_variables_to_use.extend([20000,20000])
	    minimum_variables.extend([0,0])
	    minimum_variables.extend([60000,60000])	

	prompt_cut = [R_Threshold_Max, Ep_Threshold_Min, Ep_Threshold_Max, DeltaR_Max]
	delayed_cut = [R_Threshold_Max, Ed_Threshold_Min_1, Ed_Threshold_Max_1, Ed_Threshold_Min_2, Ed_Threshold_Max_2]

	# Preprocessing of data	
	from DataProcessor import DataPreprocessor
	from DataProcessor import NeuralNetworkManager
	data_preprocessor = DataPreprocessor(InputDataset_Sig, InputDataset_Bkg, variables, variables_to_use)
	nn_manager = NeuralNetworkManager(data_preprocessor)

	#IBD cuts efficiency/purity calculation

	if(method == "BDT"):
		from TrainingManager import TrainingManager_BDT
		train, test, test_sig, test_bkg, validation, label_train, label_test, label_test_1, label_validation = nn_manager.prepare_train_test_validation()

		manager_BDT = TrainingManager_BDT(variables, training, epochs, name)
		manager_BDT.train_model_BDT(train, label_train, validation, label_validation, test, label_test)
    
		n_estimators = manager_BDT.n_estimators
		model = manager_BDT.model
		bdt_err_matrix = manager_BDT.bdt_err_matrix
		bdt_train_err_matrix = manager_BDT.bdt_train_err_matrix
		#somma_err = manager_BDT.somma_err
		#somma_train_err = manager_BDT.somma_train_err
		
		probability = model.predict_proba(test)
	
		prediction_label = np.zeros(shape = (len(probability), 1))
		for i in range (0, len(probability)):
			prediction_label[i] = probability[i][0]
		
		fpr_keras, tpr_keras, thresholds_keras = roc_curve(label_test[:, 0], prediction_label, pos_label = 1) #ora 1 è segnale e 0 è fondo
		meta = np.where(thresholds_keras < 0.501)[0][0]
		eff = tpr_keras[meta]
		pur = 1. - fpr_keras[meta]
	
		from sklearn.metrics import auc
		auc_keras = auc(fpr_keras, tpr_keras)

		from plots import plot_mean_model_history_bdt
		plot_mean_model_history_bdt(bdt_err_matrix, bdt_train_err_matrix, path, n_estimators)

		from IBD_Std_Cuts import IBD_Std_Cuts
		tpr_cuts, fpr_cuts = IBD_Std_Cuts(test_sig, test_bkg, prompt_cut, delayed_cut)

		from plots import plot_roc_curve_bdt
		plot_roc_curve_bdt(path, tpr_keras, fpr_keras, eff, pur, tpr_cuts, fpr_cuts, auc_keras)





	# Plot history for loss and accuracy
	
		
	if(method == "NN"):

		nn_manager = NeuralNetworkManager(data_preprocessor)
		train, test, test_sig, test_bkg, validation, label_train, label_test, label_test_1, label_validation = nn_manager.prepare_train_test_validation()

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
		model.load_weights(name + "/model_" + name + ".h." + str(num_best_model))

		from decision_boundaries_plotter import DecisionBoundariesPlotter
		from itertools import combinations
	
		for arg1, arg2 in combinations(variables_to_use, 2):
			decision_boundaries_plotter = DecisionBoundariesPlotter(model, arg1, arg2, variables_to_use, projection_values_variables_to_use, minimum_variables, maximum_variables, variables, 	path)
			decision_boundaries_plotter.run()

		if(All2DPlots == "yes"):
			for i in range (0,training):
				blank = path
				model.load_weights(name + "/model_" + name + ".h." + str(i))
				path = path + str(i) + ".pdf"
				decision_boundaries_plotter = DecisionBoundariesPlotter(model, "ep", "deltar", variables_to_use, projection_values_variables_to_use, minimum_variables, maximum_variables, variables, path)
				decision_boundaries_plotter.run()
				path = blank

	print("##### END #####")
	end_time = time.time()
	duration_in_seconds = end_time - start_time
	hours, remainder = divmod(duration_in_seconds, 3600)
	minutes, seconds = divmod(remainder, 60)
	print(f"Duration time: {int(hours)} h {int(minutes)} min {int(seconds)} s.")	

if __name__ == '__main__':
    main()
