def plot_mean_model_history(mean_acc, mean_val_acc, mean_loss, mean_val_loss, path):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(mean_acc)
    plt.plot(mean_val_acc)
    plt.ylabel('Mean model accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='best')
    # plt.ylim(0.97, 1)
    # plt.xlim(500, 600)
    plt.savefig(path + "_MeanModelAccuracy_History.pdf")
    plt.close()

    plt.figure(figsize=(8, 6))
    # summarize history for loss
    plt.plot(mean_loss)
    plt.plot(mean_val_loss)
    plt.ylabel('Mean model loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='best')
    plt.savefig(path + "_MeanModelLoss_History.pdf")
    plt.close()

def plot_mean_model_history_bdt(bdt_err_matrix, bdt_train_err_matrix, path, n_estimators):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    plt.figure(figsize=(8, 6))

    plt.plot(
    	np.arange(n_estimators) + 1,
    	bdt_train_err_matrix,
    	label = "Train BDT2"
	)

    plt.plot(
    	np.arange(n_estimators) + 1,
    	bdt_err_matrix,
    	label = "Validation BDT2"
    )

    plt.xlabel("Iterations")
    plt.ylabel("Error rate")

    plt.legend(loc="upper right")
    plt.savefig(path + "_BDT_history.pdf")
    plt.close()
   

def plot_roc_curve(training, path, tpr_list, fpr_list, mean_auc, dev_auc, mean_eff, dev_eff, mean_pur, dev_pur, ar, purity, tpr_cuts, fpr_cuts):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for i in range(0, training):
        if i != 0:
            plt.plot(tpr_list[i], 1. - fpr_list[i], color='blue', alpha=0.4)
        else:
            plt.plot(tpr_list[i], 1. - fpr_list[i], color='blue', alpha=0.4,
                     label='Mean AUC = {:.5f}'.format(mean_auc) + f' $\pm$ ' + '{:.5f}'.format(dev_auc))

    plt.errorbar(mean_eff, mean_pur, xerr=dev_eff, yerr=dev_pur, color='red', fmt='o',
                 label='Mean eff and purity at threshold 0.5')
    plt.text(mean_eff, mean_pur, '({:.3f}, {:.3f})'.format(mean_eff, mean_pur), ha='left')
    plt.plot(ar, purity, color='red')

    print("Average efficiency: {:.4f} +- {:.4f}".format(mean_eff, dev_eff))
    print("Average purity: {:.4f} +- {:.4f}".format(mean_pur, dev_pur))
    print("Average AUC: {:.4f} +- {:.4f}".format(mean_auc, dev_auc))

    plt.scatter(tpr_cuts, 1. - fpr_cuts, color='black', marker='o', label='Canonical IBD cuts')
    plt.text(tpr_cuts, 1. - fpr_cuts, '({:.3f}, {:.3f})'.format(tpr_cuts, 1. - fpr_cuts), ha='left')

    plt.xlabel('Selection efficiency (= true positive rate)')
    plt.ylabel('Signal purity (= 1 - false positive rate)')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(path + "ROC_Curve.pdf")
    plt.grid()
    plt.show()

def plot_roc_curve_bdt(path, tpr_keras, fpr_keras, eff, pur, tpr_cuts, fpr_cuts, auc_keras):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.plot(tpr_keras, 1. - fpr_keras, label='AUC = {:.5f}'.format(auc_keras))
    plt.scatter(eff, pur, color = 'red', marker = 'o', label = 'Eff and pur at threshold 0.5')
    plt.text(eff, pur, '({:.3f}, {:.3f})'.format(eff, pur), ha='left')
    plt.legend()

    plt.scatter(tpr_cuts, 1. - fpr_cuts, color='black', marker='o', label='Canonical IBD cuts')
    plt.text(tpr_cuts, 1. - fpr_cuts, '({:.3f}, {:.3f})'.format(tpr_cuts, 1. - fpr_cuts), ha='left')

    plt.xlabel('Selection efficiency (= true positive rate)')
    plt.ylabel('Signal purity (= 1 - false positive rate)')
    plt.savefig(path + "ROC_Curve_bdt.pdf")
    plt.grid()
    plt.show()

def correlation_matrix_sig_bkg(df_sig_histo,df_bkg_histo,path):
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	correlation_matrix_sig = df_sig_histo.corr()
	sns.set(style="white")
	plt.figure(figsize=(8, 6))
	sns.heatmap(correlation_matrix_sig, annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # Specifica la scala di colore e i valori min/max
	plt.title("Correlation Matrix - Resampled Signal")
	plt.savefig(path+"_CorrelationMatrix_Sig.pdf")
	plt.close()
	
	correlation_matrix_bkg = df_bkg_histo.corr()
	sns.set(style="white")
	plt.figure(figsize=(8, 6))
	sns.heatmap(correlation_matrix_bkg, annot=True, cmap='coolwarm', vmin=-1, vmax=1)  # Specifica la scala di colore e i valori min/max
	plt.title("Correlation Matrix - Resampled Background")
	plt.savefig(path+"_CorrelationMatrix_Bkg.pdf")
	plt.close()
	
def plots_variables_1D_distributions(histo_data,variables,path):	
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns

	fig, ax = plt.subplots(4, 4, figsize=(20,25))
	Bins_OutputPlots = 100
	
	ax[0,0] = sns.histplot(histo_data, x = "rp3", log_scale = (False, True), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[0,0])
	#ax[0,0].set_title(f"$r^3$ prompt - Prediction of the NN")
	ax[0,0].set_xlabel(f"$r^3$ [m$^3$]")
	ax[0,0].set_ylim(1, 3e05)
	ax[0,0].grid()
	
	ax[0,1] = sns.histplot(histo_data, x = "rp3", log_scale = (False, True), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[0,1])
	#ax[0,1].set_title(f"$r^3$ prompt - Truth")
	ax[0,1].set_xlabel(f"$r^3$ [m$^3$]")
	ax[0,1].set_ylim(1, 3e05)
	plt.legend(["Background", "Signal"], loc = 'best')
	ax[0,1].grid()
	
	ax[0,2] = sns.histplot(histo_data, x = "rd3", log_scale = (False, True), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[0,2])
	#ax[0,2].set_title(f"$r^3$ delayed - Prediction of the NN")
	ax[0,2].set_xlabel(f"$r^3$ [m$^3$]")
	ax[0,2].set_ylim(1, 3e05)
	ax[0,2].grid()
	
	ax[0,3] = sns.histplot(histo_data, x = "rd3", log_scale = (False, True), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[0,3])
	#ax[0,3].set_title(f"$r^3$ delayed - Truth")
	ax[0,3].set_xlabel(f"$r^3$ [m$^3$]")
	plt.legend(["Background", "Signal"], loc = 'best')
	ax[0,3].set_ylim(1, 3e05)
	ax[0,3].grid()
	
	ax[1,0] = sns.histplot(histo_data, x = "ep", log_scale = (False, True), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[1,0], binrange = (0, 12))
	#ax[1,0].set_title(f"Energy prompt - Prediction of the NN")
	ax[1,0].set_xlabel(f"E [MeV]")
	ax[1,0].grid()
	
	ax[1,1] = sns.histplot(histo_data, x = "ep", log_scale = (False, True), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[1,1], binrange = (0, 12))
	#ax[1,1].set_title(f"Energy prompt - Truth")
	ax[1,1].set_xlabel(f"E [MeV]")
	plt.legend(["Background", "Signal"], loc = 'best')
	ax[1,1].grid()
	
	ax[1,2] = sns.histplot(histo_data, x = "ed", log_scale = (False, True), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[1,2], binrange = (0, 12))
	#ax[1,2].set_title(f"Energy delayed - Prediction of the NN")
	ax[1,2].set_xlabel(f"E [MeV]")
	ax[1,2].grid()
	
	ax[1,3] = sns.histplot(histo_data, x = "ed", log_scale = (False, True), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[1,3], binrange = (0, 12))
	#ax[1,3].set_title(f"Energy delayed - Truth")
	plt.legend(["Background", "Signal"], loc = 'best')
	ax[1,3].set_xlabel(f"E [MeV]")
	ax[1,3].grid()
	
	ax[2,0] = sns.histplot(histo_data, x = "deltar", log_scale = (False, True), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[2,0], binrange = (0, 3))
	#ax[2,0].set_title(f"$\Delta r$ - Prediction of the NN")
	ax[2,0].set_xlabel(f"$\Delta r$ [m]")
	ax[2,0].set_ylim(1, 200000)
	ax[2,0].grid()
	
	ax[2,1] = sns.histplot(histo_data, x = "deltar", log_scale = (False, True), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[2,1], binrange = (0, 3))
	#ax[2,1].set_title(f"$\Delta r$ - Truth")
	ax[2,1].set_xlabel(f"$\Delta r$ [m]")
	#ax[2,1].set_ylim(1, 200000)
	ax[2,1].grid()
	
	if(variables == 7):
		ax[2,2] = sns.histplot(histo_data, x = "QLpFlat", log_scale = (False, False), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[2,2], binrange = (4000, 90000))
		#ax[2,2].set_title(f"m_QLp - Prediction of the NN")
		ax[2,2].set_xlabel(f"m_QLp")
		#ax.set_ylim(1, 200000)
		ax[2,2].grid()
	
		ax[2,3] = sns.histplot(histo_data, x = "QLpFlat", log_scale = (False, False), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[2,3], binrange = (4000, 90000))
		#ax[2,3].set_title(f"m_QLp - Truth")
		ax[2,3].set_xlabel(f"m_QLp")
		plt.legend(["Background", "Signal"], loc = 'best')
		#ax.set_ylim(1, 200000)
		ax[2,3].grid()
	
		ax[3,0] = sns.histplot(histo_data, x = "QLdFlat", log_scale = (False, False), hue = "prediction", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[3,0], binrange = (4000, 90000))
		ax[3,0].set_title(f"m_QLd - Prediction of the NN")
		ax[3,0].set_xlabel(f"m_QLd")
		#ax.set_ylim(1, 200000)
		ax[3,0].grid()
	
		ax[3,1] = sns.histplot(histo_data, x = "QLdFlat", log_scale = (False, False), hue = "truth", palette = ['blue', 'red'], bins = Bins_OutputPlots, legend = True, ax = ax[3,1], binrange = (4000, 90000))
		ax[3,1].set_title(f"m_QLd - Truth")
		ax[3,1].set_xlabel(f"m_QLd")
		plt.legend(["Background", "Signal"], loc = 'best')
		#ax.set_ylim(1, 200000)
		ax[3,1].grid()
	
	plt.savefig(path + "_Outputs.pdf")

