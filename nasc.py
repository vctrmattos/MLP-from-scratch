#Activation and loss functions and their derivatives

e  = 2.7182818284590452

def sigmoid(x):
    if abs(x)>99:
        return 0
    else:
        return 1/(1 + e**(-x))

def relu(x):
    return max(0, x)

def tanh(x):
    if x > 700:
        return 1
    elif x < -700:
        return -1
    return (e**x - e**(-x))/(e**x + e**(-x))

def step(x):
    return 1 if x >= 0 else 0

def sigmoid_d(x):
    s = sigmoid(x)
    return (1 - s)*s

def tanh_d(x):
    return 1 - (tanh(x))**2

def mse(y_true, y_pred):
    return (((y_true-y_pred)**2)).mean()

def mse_d(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size