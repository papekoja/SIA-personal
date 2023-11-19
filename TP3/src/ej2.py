import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import perceptron
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted)**2)

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def plot_learning_curve(training_error_history):
    plt.plot(training_error_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Learning Curve')
    plt.show()
    
# Read the CSV file into a DataFrame
df = pd.read_csv('sia-tp3/data/TP3-ej2-conjunto.csv')

# Normalize the features (x1, x2, x3)
feature_cols = ['x1', 'x2', 'x3']
df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())

# Normalize the target (y)
df['y'] = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())

# pt create learning rate graphs
""" for i in range(1, 11):
    p = perceptron.Perceptron(0.001*i, 1)
    p.non_linear = True
    p.train(df_train, epochs=100)
    plt.plot(range(1, len(p.training_error_history) + 1), p.training_error_history, label=f"Learning rate = {0.001*i}")

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Error over Epochs for Different Learning Rates')
plt.legend()    
plt.show() """

# create beta 3d graph

""" # Train the perceptron with varying learning rates and beta values
learning_rates = np.linspace(0.001, 0.1, 10)  # for example
beta_values = np.linspace(0.5, 5, 10)        # for example
errors = np.zeros((len(learning_rates), len(beta_values)))

for i, lr in enumerate(learning_rates):
    for j, beta in enumerate(beta_values):
        p = perceptron.Perceptron(lr, beta)
        p.non_linear = True
        p.learning_rate = lr
        p.train(df_train, epochs=100)
        # calculate the average error from error history
        #error = np.mean(p.training_error_history)
        #calculate the mean squared error from test data
        y_pred = p.output(df_test.iloc[:, 0].values, df_test.iloc[:, 1].values, df_test.iloc[:, 2].values)
        error = mean_squared_error(df_test.iloc[:, 3].values, y_pred)
        errors[i, j] = error

# Plot the results on a 3D graph
X, Y = np.meshgrid(learning_rates, beta_values)
Z = errors.T  # transpose the error matrix to align with meshgrid

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel('Learning Rate')
ax.set_ylabel('Beta Value')
ax.set_zlabel('Error')
ax.set_title('Error for different Learning Rates and Beta Values')

plt.show() """

# 2.2 Compare training and test error for different test sizes

# Split the data into training and test sets

test_sizes = np.linspace(0.1, 0.9, 9)
training_errors = []
test_errors = []

for test_size in test_sizes:
    df_train, df_test = train_test_split(df, test_size=test_size)
    p = perceptron.Perceptron(0.06, 3)
    p.non_linear = True
    p.train(df_train, epochs=100)
    training_errors.append(p.training_error_history[-1])
    y_pred = p.output(df_test.iloc[:, 0].values, df_test.iloc[:, 1].values, df_test.iloc[:, 2].values)
    test_errors.append(mean_squared_error(df_test.iloc[:, 3].values, y_pred))

plt.plot(test_sizes, training_errors, label='Training Error')
plt.plot(test_sizes, test_errors, label='Test Error')
plt.xlabel('Test Size')
plt.ylabel('MSE')
plt.title('Training and Test Error for Different Test Sizes')
plt.legend()
plt.show()
