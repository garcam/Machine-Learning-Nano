import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = target - output

# TODO: Calculate error term for output layer
output_error_term = error * output * (1 - output)

# TODO: Calculate error term for hidden layer
hidden_error_term = np.dot(output_error_term, weights_hidden_output) * \
                    hidden_layer_output * (1 - hidden_layer_output)

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = learnrate * output_error_term * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = learnrate * hidden_error_term * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)




#############################
####### otro
###########################
# este es el binary.csv:

admit,gre,gpa,rank
0,380,3.61,3
1,660,3.67,3
1,800,4,1
1,640,3.19,4
0,520,2.93,4
1,760,3,2
1,560,2.98,1
0,400,3.08,2
1,540,3.39,3
0,700,3.92,2
0,800,4,4
0,440,3.22,1
1,760,4,1
0,700,3.08,2
1,700,4,1
0,480,3.44,3
0,780,3.87,4
0,360,2.56,3
0,800,3.75,2
1,540,3.81,1
0,500,3.17,3
1,660,3.63,2
0,600,2.82,4
0,680,3.19,4
1,760,3.35,2
1,800,3.66,1
1,620,3.61,1
1,520,3.74,4
1,780,3.22,2
0,520,3.29,1
0,540,3.78,4
0,760,3.35,3
0,600,3.4,3
1,800,4,3
0,360,3.14,1
0,400,3.05,2
0,580,3.25,1
0,520,2.9,3
1,500,3.13,2
1,520,2.68,3
0,560,2.42,2
1,580,3.32,2
1,600,3.15,2
0,500,3.31,3
0,700,2.94,2
1,460,3.45,3
1,580,3.46,2
0,500,2.97,4
0,440,2.48,4
0,400,3.35,3
0,640,3.86,3
0,440,3.13,4
0,740,3.37,4
1,680,3.27,2
0,660,3.34,3
1,740,4,3
0,560,3.19,3
0,380,2.94,3
0,400,3.65,2
0,600,2.82,4
1,620,3.18,2
0,560,3.32,4
0,640,3.67,3
1,680,3.85,3
0,580,4,3
0,600,3.59,2
0,740,3.62,4
0,620,3.3,1
0,580,3.69,1
0,800,3.73,1
0,640,4,3
0,300,2.92,4
0,480,3.39,4
0,580,4,2
0,720,3.45,4
0,720,4,3
0,560,3.36,3
1,800,4,3
0,540,3.12,1
1,620,4,1
0,700,2.9,4
0,620,3.07,2
0,500,2.71,2
0,380,2.91,4
1,500,3.6,3
0,520,2.98,2
0,600,3.32,2
0,600,3.48,2
0,700,3.28,1
1,660,4,2
0,700,3.83,2
1,720,3.64,1
0,800,3.9,2
0,580,2.93,2
1,660,3.44,2
0,660,3.33,2
0,640,3.52,4
0,480,3.57,2
0,700,2.88,2
0,400,3.31,3
0,340,3.15,3
0,580,3.57,3
0,380,3.33,4
0,540,3.94,3
1,660,3.95,2
1,740,2.97,2
1,700,3.56,1
0,480,3.13,2
0,400,2.93,3
0,480,3.45,2
0,680,3.08,4
0,420,3.41,4
0,360,3,3
0,600,3.22,1
0,720,3.84,3
0,620,3.99,3
1,440,3.45,2
0,700,3.72,2
1,800,3.7,1
0,340,2.92,3
1,520,3.74,2
1,480,2.67,2
0,520,2.85,3
0,500,2.98,3
0,720,3.88,3
0,540,3.38,4
1,600,3.54,1
0,740,3.74,4
0,540,3.19,2
0,460,3.15,4
1,620,3.17,2
0,640,2.79,2
0,580,3.4,2
0,500,3.08,3
0,560,2.95,2
0,500,3.57,3
0,560,3.33,4
0,700,4,3
0,620,3.4,2
1,600,3.58,1
0,640,3.93,2
1,700,3.52,4
0,620,3.94,4
0,580,3.4,3
0,580,3.4,4
0,380,3.43,3
0,480,3.4,2
0,560,2.71,3
1,480,2.91,1
0,740,3.31,1
1,800,3.74,1
0,400,3.38,2
1,640,3.94,2
0,580,3.46,3
0,620,3.69,3
1,580,2.86,4
0,560,2.52,2
1,480,3.58,1
0,660,3.49,2
0,700,3.82,3
0,600,3.13,2
0,640,3.5,2
1,700,3.56,2
0,520,2.73,2
0,580,3.3,2
0,700,4,1
0,440,3.24,4
0,720,3.77,3
0,500,4,3
0,600,3.62,3
0,400,3.51,3
0,540,2.81,3
0,680,3.48,3
1,800,3.43,2
0,500,3.53,4
1,620,3.37,2
0,520,2.62,2
1,620,3.23,3
0,620,3.33,3
0,300,3.01,3
0,620,3.78,3
0,500,3.88,4
0,700,4,2
1,540,3.84,2
0,500,2.79,4
0,800,3.6,2
0,560,3.61,3
0,580,2.88,2
0,560,3.07,2
0,500,3.35,2
1,640,2.94,2
0,800,3.54,3
0,640,3.76,3
0,380,3.59,4
1,600,3.47,2
0,560,3.59,2
0,660,3.07,3
1,400,3.23,4
0,600,3.63,3
0,580,3.77,4
0,800,3.31,3
1,580,3.2,2
1,700,4,1
0,420,3.92,4
1,600,3.89,1
1,780,3.8,3
0,740,3.54,1
1,640,3.63,1
0,540,3.16,3
0,580,3.5,2
0,740,3.34,4
0,580,3.02,2
0,460,2.87,2
0,640,3.38,3
1,600,3.56,2
1,660,2.91,3
0,340,2.9,1
1,460,3.64,1
0,460,2.98,1
1,560,3.59,2
0,540,3.28,3
0,680,3.99,3
1,480,3.02,1
0,800,3.47,3
0,800,2.9,2
1,720,3.5,3
0,620,3.58,2
0,540,3.02,4
0,480,3.43,2
1,720,3.42,2
0,580,3.29,4
0,600,3.28,3
0,380,3.38,2
0,420,2.67,3
1,800,3.53,1
0,620,3.05,2
1,660,3.49,2
0,480,4,2
0,500,2.86,4
0,700,3.45,3
0,440,2.76,2
1,520,3.81,1
1,680,2.96,3
0,620,3.22,2
0,540,3.04,1
0,800,3.91,3
0,680,3.34,2
0,440,3.17,2
0,680,3.64,3
0,640,3.73,3
0,660,3.31,4
0,620,3.21,4
1,520,4,2
1,540,3.55,4
1,740,3.52,4
0,640,3.35,3
1,520,3.3,2
1,620,3.95,3
0,520,3.51,2
0,640,3.81,2
0,680,3.11,2
0,440,3.15,2
1,520,3.19,3
1,620,3.95,3
1,520,3.9,3
0,380,3.34,3
0,560,3.24,4
1,600,3.64,3
1,680,3.46,2
0,500,2.81,3
1,640,3.95,2
0,540,3.33,3
1,680,3.67,2
0,660,3.32,1
0,520,3.12,2
1,600,2.98,2
0,460,3.77,3
1,580,3.58,1
1,680,3,4
1,660,3.14,2
0,660,3.94,2
0,360,3.27,3
0,660,3.45,4
0,520,3.1,4
1,440,3.39,2
0,600,3.31,4
1,800,3.22,1
1,660,3.7,4
0,800,3.15,4
0,420,2.26,4
1,620,3.45,2
0,800,2.78,2
0,680,3.7,2
0,800,3.97,1
0,480,2.55,1
0,520,3.25,3
0,560,3.16,1
0,460,3.07,2
0,540,3.5,2
0,720,3.4,3
0,640,3.3,2
1,660,3.6,3
1,400,3.15,2
1,680,3.98,2
0,220,2.83,3
0,580,3.46,4
1,540,3.17,1
0,580,3.51,2
0,540,3.13,2
0,440,2.98,3
0,560,4,3
0,660,3.67,2
0,660,3.77,3
1,520,3.65,4
0,540,3.46,4
1,300,2.84,2
1,340,3,2
1,780,3.63,4
1,480,3.71,4
0,540,3.28,1
0,460,3.14,3
0,460,3.58,2
0,500,3.01,4
0,420,2.69,2
0,520,2.7,3
0,680,3.9,1
0,680,3.31,2
1,560,3.48,2
0,580,3.34,2
0,500,2.93,4
0,740,4,3
0,660,3.59,3
0,420,2.96,1
0,560,3.43,3
1,460,3.64,3
1,620,3.71,1
0,520,3.15,3
0,620,3.09,4
0,540,3.2,1
1,660,3.47,3
0,500,3.23,4
1,560,2.65,3
0,500,3.95,4
0,580,3.06,2
0,520,3.35,3
0,500,3.03,3
0,600,3.35,2
0,580,3.8,2
0,400,3.36,2
0,620,2.85,2
1,780,4,2
0,620,3.43,3
1,580,3.12,3
0,700,3.52,2
1,540,3.78,2
1,760,2.81,1
0,700,3.27,2
0,720,3.31,1
1,560,3.69,3
0,720,3.94,3
1,520,4,1
1,540,3.49,1
0,680,3.14,2
0,460,3.44,2
1,560,3.36,1
0,480,2.78,3
0,460,2.93,3
0,620,3.63,3
0,580,4,1
0,800,3.89,2
1,540,3.77,2
1,680,3.76,3
1,680,2.42,1
1,620,3.37,1
0,560,3.78,2
0,560,3.49,4
0,620,3.63,2
1,800,4,2
0,640,3.12,3
0,540,2.7,2
0,700,3.65,2
1,540,3.49,2
0,540,3.51,2
0,660,4,1
1,480,2.62,2
0,420,3.02,1
1,740,3.86,2
0,580,3.36,2
0,640,3.17,2
0,640,3.51,2
1,800,3.05,2
1,660,3.88,2
1,600,3.38,3
1,620,3.75,2
1,460,3.99,3
0,620,4,2
0,560,3.04,3
0,460,2.63,2
0,700,3.65,2
0,600,3.89,3

import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std
    
# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))




