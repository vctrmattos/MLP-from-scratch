from nasc import *

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_dict = {
            mse: mse_d
        }

    def add_layer(self, layer):
        self.layers.append(layer)
    
    def predict(self, input_data):
        # sample dimension first
        samples = input_data.rows
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.extend(output.array)
        return result
    
    # Calculate accuracy percentage between two lists


    def set_loss(self, loss):
        self.loss = loss
        self.loss_d = self.loss_dict[self.loss]

    # train the network
    def fit_and_test(self, x_train, y_train, x_val, y_val, epochs, learning_rate, acc_fun=None, progression=False, iter_step=5):
        # sample dimension first
        samples = x_train.rows
        params = {
                    "error":[], 
                    "acc_train":[], 
                    "acc_val":[],
                    "epoch" : []
                 }
        cont = cont2 = 0
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                
                # Alpha dinamico --> self.loss(y_train[j], output)

                err += self.loss(y_train[j], output)
                

                # backward propagation
                error = self.loss_d(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
                    
                    if progression and not (cont % iter_step):
                        #cont+=1
                        y_predicted = self.predict(x_train)
                        acc_train = acc_fun(y_train, y_predicted)

                        y_val_predicted = self.predict(x_val)
                        acc_val  = acc_fun(y_val, y_val_predicted)
                        
                        # params["acc_train"].append(acc_train)
                        # params["acc_val"].append(acc_val)
                        # params["epoch"].append(i + 1)
                        # params["error"].append(err)

            # calculate average error on all samples
            err /= samples
            cont += 1 

            if not (i % iter_step) and not progression: #tomar cuidado com isso aqui, pode mudar o valor final mostrado
                print(f"epoch {i + 1}/{epochs}   error={err:.5}")
            elif not (i % iter_step):

                params["acc_train"].append(acc_train)
                params["acc_val"].append(acc_val)
                params["epoch"].append(i + 1)
                params["error"].append(err)
                print(f"epoch {i + 1}/{epochs} error={err:.5} acc_val={params['acc_val'][cont2]} acc_train={params['acc_train'][cont2]}")
                cont2 += 1
        
        return params