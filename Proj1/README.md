# Project 1: Classification, weight sharing, auxiliary losses

The experiments from the part II of our report (first class of models) are contained in `Experiences_Part_1.ipynb`.  
The experiments from the part III of our report (second class of models) are contained in `Experiences_Part_2.ipynb`.  
To increase the readability of these files we used helper function, located in `utils.py` (data loading and preprocessing, console printing, plot functions), `utils_pipeline1.py` (test, train, cross-validation and evaluation for models of the first class) and `utils_pipeline2.py` (test, train, cross-validation and evaluation for models of the second class). All of the models (first & second class) are defined in `models.py`. The original code from class to load the dataset is in `dlc_practical_prologue.py`.  
  
`test.py` contains a script that creates our model with the best accuracy (Siamese network with 3 convolutional layers), and trains it with the best hyperparameters that we found (aux_alpha_loss = 0.4). It does so 10 independent times and prints the mean test accuracy and its standard deviation.
