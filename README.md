# IBD_Selection_ML

Selection cuts for IBD analysis with Machine Learning (Neural Network, Boosted Decision Trees) methods

Authors: Davide Basilico (davide.basilico@mi.infn.it), Gloria Senatore (gloria.senatore@mi.infn.it)

Other information: https://docs.google.com/presentation/d/1hnqCs1f93pJPCweK5U3vMBuM5VKYMLdVKGHwifIv1-I/edit?usp=sharing

-- How to launch:
python3 IBD_Selection_ML.py --config CfgFile.cfg

-- Exemplary CfgFile.cfg:
name = Std (name of the outputs; a folder named "name" will be also created)

epochs = 10

training = 10

variables = 5  (or 7, if one wants to include likelihood/E^alpha variables for prompt and delayed)

InputDataset_Sig = InputDataset_Sig.txt

InputDataset_Bkg = InputDataset_Bkg.txt

R_Threshold_Max = 17.2

DeltaR_Max = 1.5

Ep_Threshold_Min = 0.7

Ep_Threshold_Max = 12

Ed_Threshold_Min_1 = 1.9

Ed_Threshold_Max_1 = 2.5

Ed_Threshold_Min_2 = 4.9

Ed_Threshold_Max_2 = 5.5

QLp_Threshold_Min = 15900

QLd_Threshold_Max = 18000

PowerLawCoefficient_p = 0.4977

PowerLawCoefficient_d = 0.4929 
