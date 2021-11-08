# Mask_Time_Dynamics

To reproduce the results given in our manuscript __Capturing Time Dynamics from Speech using Neural Networks for Surgical Masks Detection__, please follow the next steps:

S1 - Use __creat_seeds.py__ to create a list of paths to the wav files given in Mask Augsburg Speech Corpus (MASC) and their labels. The created seeds (in .pkl format) can be found in folder __seeds__. You can then check the distribution of the data for each split, i.e., train, validation and test set, as well as the gender information. The results should correspond to TABLE I in manuscript.

S2 - Run __train.py__ for training, you can select the models by changing type of conv_net and recurrent_net. Taking convolutional transformer network, which is one mainly proposed architecture, conv_net is set to "cnn", while recurrent_net is set to "tx".

S3 - Run __evaluate.py__ for testing.

S4 - (optional) MTL indicate multi-task learning, where the gender of a speaker and whether she/he is wearing a surgical mask can be recognised simultaneously in one network. The evaluation script can be used as S3 after training the MTL network.

S5 - (optional) We also provide the code for results analysis and visualisation, such as __plot_roc_curve__ which works based on the evaluation results (Fig. 3 in manuscript). Note that you will need to change the path to your own saved results. 

The diagram, trained models as well as citation links will be prepared and added later. 
