import numpy as np

# Function to initialize parameters for the neural network
def initialize_parameters(layer_sizes):
    parameters = {}  # Storing it in an empty dictionary

    for i in range(1, len(layer_sizes)):
        # Initializing weights with He initialization (good for ReLU activation)
        parameters[f'w{i}'] = np.random.randn(layer_sizes[i], layer_sizes[i-1]) * np.sqrt(2 / layer_sizes[i-1])
        
        # Initializing biases as zero column vectors
        parameters[f'b{i}'] = np.zeros((layer_sizes[i], 1))  
    
    return parameters  # Returning the initialized parameters

# Function to apply activation functions
def apply_activation(z, activation_function):
    if activation_function == "relu":
        A = np.maximum(0, z)  # Apply ReLU function
    
    elif activation_function == "sigmoid":
        A = 1 / (1 + np.exp(-z))  # Apply Sigmoid function
        
    elif activation_function == "softmax":
        soft_A = np.exp(z - np.max(z, axis=0, keepdims=True))  # Apply exponentials with stability
        sum_A = np.sum(soft_A, axis=0, keepdims=True)  # Sum across columns
        A = soft_A / sum_A  # Normalize for softmax output
    
    else:
        print("Unknown activation function")  # Error handling for invalid activation functions
        return None
    
    return A  # Returning the activated values

# Function for forward propagation
def forward_propagation(X, parameters, activation_functions):
    A = X  # Input is the first activation
    activations = {"A0": X}  # Store all activations
    L = len(parameters) // 2  # Number of layers

    for i in range(1, L + 1):  # Hidden layers and output layer
        Z = np.dot(parameters[f'w{i}'], A) + parameters[f'b{i}']
        A = apply_activation(Z, activation_function=activation_functions[i-1])
        activations[f"A{i}"] = A  # Store activation

    return A, activations  # Return final output and activations

# Function to compute categorical cross-entropy loss
def compute_loss(Y_true, Y_pred):
    m = Y_true.shape[1]  # Number of samples
    loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m  # Adding small value to avoid log(0)
    return loss

# Function for backpropagation
def backpropagation(X, Y_true, Y_pred, parameters, activations, activation_functions):
    grads = {}  # Dictionary to store gradients
    L = len(parameters) // 2  # Number of layers
    m = X.shape[1]  # Number of training samples

    # Compute dZ for the output layer (Softmax + Cross-Entropy)
    dZ = Y_pred - Y_true  

    # Loop through layers in reverse order
    for i in reversed(range(1, L + 1)):
        A_prev = activations[f"A{i-1}"]  # Use stored activation

        # Compute gradients
        grads[f'dw{i}'] = (1/m) * np.dot(dZ, A_prev.T)  # Ensure alignment
        grads[f'db{i}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)  # Sum over batch size

        # Compute dA_prev for next layer
        if i > 1:  # No need to compute for input layer
            dA_prev = np.dot(parameters[f"w{i}"].T, dZ)

            # Apply ReLU derivative for backpropagation
            dZ = dA_prev * (A_prev > 0)  

    return grads  # Return the computed gradients


# Example usage: Define neural network structure
layer_sizes = (784, 256, 128, 10)  # Input layer -> Hidden1 -> Hidden2 -> Output layer
parameters = initialize_parameters(layer_sizes)  # Initialize parameters

# Define activation functions for each layer
activation_functions = ["relu", "relu", "softmax"]

# Simulated single input sample with 784 features
X_sample = np.random.randn(784, 3)  # Simulating 3 samples

# Perform forward propagation
Y_pred, activations = forward_propagation(X_sample, parameters, activation_functions)
print("Output shape:", Y_pred.shape)  # Should be (10, 3), matching the output layer size

# Simulating true labels (one-hot encoding for 3 samples)
Y_true = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Class 0
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Class 1
                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]).T  # Class 5

# Simulated softmax predictions (each column sums to 1)
Y_pred_simulated = np.array([[0.9, 0.05, 0.02],  
                              [0.05, 0.8, 0.02],  
                              [0.02, 0.05, 0.02],  
                              [0.01, 0.02, 0.02],  
                              [0.01, 0.02, 0.02],  
                              [0.01, 0.02, 0.9],  
                              [0.01, 0.02, 0.02],  
                              [0.01, 0.02, 0.02],  
                              [0.01, 0.02, 0.02],  
                              [0.01, 0.02, 0.02]])

# Compute loss
loss = compute_loss(Y_true, Y_pred_simulated)
print("Loss:", loss)

# Compute gradients using backpropagation
gradients = backpropagation(X_sample, Y_true, Y_pred, parameters, activations, activation_functions)
print("Gradients computed successfully!")

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  # Number of layers
    
    for i in range(1, L + 1):
        parameters[f'w{i}'] -= learning_rate * grads[f'dw{i}']
        parameters[f'b{i}'] -= learning_rate * grads[f'db{i}']
    
    return parameters

# Hyperparameters
epochs = 1000  # Number of training iterations
learning_rate = 0.01  # Step size
loss_history = []  # Store loss over epochs

for epoch in range(1000):  # Example: 1000 epochs
    # Forward pass
    Y_pred, activations = forward_propagation(X_sample, parameters, activation_functions)

    # Compute loss
    loss = compute_loss(Y_true, Y_pred)
    loss_history.append(loss)  # Store loss

    # Backpropagation
    gradients = backpropagation(X_sample, Y_true, Y_pred, parameters, activations, activation_functions)

    # Update parameters (use a small learning rate)
    learning_rate = 0.01
    for i in range(1, len(parameters) // 2 + 1):
        parameters[f'w{i}'] -= learning_rate * gradients[f'dw{i}']
        parameters[f'b{i}'] -= learning_rate * gradients[f'db{i}']

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot the loss history
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
