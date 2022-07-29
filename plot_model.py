import seaborn as sns
import matplotlib.pyplot as plt

def plot_model(params):
    '''
    Plot model parameters.
    '''
    acc_train = params["acc_train"]
    acc_val = params["acc_val"] 
    epoch = params["epoch"]
    error = params["error"]

    # plotting the line 1 points
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    sns.lineplot(x = epoch, y = acc_train, label = "Train")
    sns.lineplot(x = epoch, y = acc_val, label = "Val")
    
    plt.title('Model Accuracy')
    
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.show()
