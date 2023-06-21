import torch
from torch import nn
# Build a simple neural network using a simple activation function
# An activation takes any real number as input, also known as its domain and outputs a number in a certain range
# using a nonlinear differentiable function


########################
# Activation Functions #
########################

# ReLU non linearity: sets all negative numbers in a tensor to 0,
#   so it means that the range for ReLU is 0 to a certain value

# Sigmoid: Gives a number between 0 and 1

# Tanh: Hyperbolic Tan

# For the activation function we're gonna use ReLU



##############
# Optimizers #
##############
# Algorithms or methods used to change the attributes of the neural network
# such as the weights and learning rate to reduce the loses (errors)
# Optimizers are used to solved optimization problems by minimizing the function

# Types of Optimizers:
#   - Gradient descent
#   - Stochastic gradient descent
#   - Mini batch gradient descent
#   - Momentum based gradient descent
#   - Adam optimizers

# Optimizers help us fine tune our neural network

linear = nn.Linear(10, 2) # linear transformation
inp = torch.randn(3, 10)
out = linear(inp)


# Create activation function
relu = nn.ReLU()
relu_output = relu(out)

# print(relu_output)

"""
Create optimizer
optim.Adam = Adam Optimizer
pass the parameters to be optimized and the learning rate (lr)

For all nn you can access its parameters as a list using the parameters function
 nn.sequential: creates a single operation that performs a sequence of operations
Create a NN

This NN will execute different operations:
- Linear transformation, followed by the Batch Norm1d and then the ReLU

The Batch Norm1d is a normalization technique that will re-scale a batch
of n inputs to have a consistent, mean and standard deviation between batches.
This helps to stabilize this NN
"""

# nn.Linear(5, 2) -> dimensions
# nn.BatchNorm1d(2) -> 2 is the dimension from the Linear
mlp_layer = nn.Sequential(nn.Linear(5, 2), nn.BatchNorm1d(2), nn.ReLU())

"""
mlp_layer
tensor([[0.0000, 0.2614],
        [0.0000, 0.0000],
        [0.0000, 0.8869]], grad_fn=<ReluBackward0>)
Sequential(
  (0): Linear(in_features=5, out_features=2, bias=True)
  (1): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()

"""

tensorInput = torch.randn(5, 5) + 1 # Produces a random tensor in which all elements will be having an additon of one to their values
mlp_layer(tensorInput)

"""
mlp_layer(tensorInput)
tensor([[0.0000, 0.0150],
        [0.0000, 0.0000],
        [0.0000, 1.6878],
        [1.9297, 0.3122],
        [0.0000, 0.0000]], grad_fn=<ReluBackward0>)
"""

# Optimize
adam_opt = torch.optim.Adam(mlp_layer.parameters(), lr=1e-1) # lr = 1e-1 = 10^⁻1

"""
print(adam_opt)

Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.1
    maximize: False
    weight_decay: 0
)
"""


##########################
# Basic ML Training Loop #
##########################

"""
A training loop consists of 4 different parts:
  1- Zero out the gradients: set all the gradients to 0 usig opt.ZeroGrad. This needed because in
    Python once we do some backward propagation, PyTorch accumulates the gradient
    on subsquent backward passes, so good practice is to always set to 0 at the
    beggining so the parameter update is done properly

  2- Calculate the loss

  3- Calculate the gradients with respect to the loss using loss.backward method

  4- Update the parameters being optimized using opt.step
"""


# For a successfull machine learning model training, a lot of loops or
# epochs are being performed. This example focus only on one single loop

# Create the tensor
train_example = torch.randn(100, 5) + 1

# Zero out the gradients
adam_opt.zero_grad()

# Calculate the loss using mean
curr_loss = torch.abs(1 - mlp_layer(train_example)).mean()

# Perform backward propagation on the loss
curr_loss.backward()

# Perform a step
adam_opt.step()

print(curr_loss)

