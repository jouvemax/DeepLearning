from modules import *
from helpers import *

#generate disk data
N = 1000
train_input, train_target, test_input, test_target = generate_disk_data(nb_points=N)

#model and criterion definition
model = Sequential(Linear(2,25), ReLU(), Linear(25,25),
                   ReLU(), Linear(25,25), ReLU(), Linear(25,2))
criterion = LossMSE()

#train the model
train_model(model, criterion, train_input, train_target, 1000, 
                100, 0.01, logging=True)

#evaluate the model
print("On test set:")
evaluate_model(model, test_input, test_target, logging=True)

