
def initialise_net():
    """
    Initialise feed forward neural network with 4 fully connected layers of size 48, 100, 70 and 55. 
    
    **Returns**:
    torch.nn
    """
    net = torch.nn.Sequential(
            torch.nn.Linear(48, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(100, 70),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(70, 55),
        )
    return 


def my_loss(y_true, y_pred):  # Mean Absolute Percentual Error (MAPE)
    """
    Given the target and the current values, compute the Mean Absolute Percentage Error (MAPE).
    
    **Args**:
    
    - y_true(array): Target value. 
    - y_pred(array): Current value. 
    
    **Returns**:
    MAPE. 
    
    """
    return 


def iterate():
    """Minimise MAPE"""
    torch.manual_seed(28)    # To keep it reproducible
    x = inputs; y = targets # Relabel input and target variables
    x, y = Variable(x), Variable(y) # torch can only train on Variable, so convert them to Variable
    plt.figure()
    for learning_rate in [0.01, 0.005, 0.0025, 0.001]: # Test different learning rates (1D line search)
        net = initialise_net() # Initialise network
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = 0.7) # Uncomment to switch to SGD optimiser
        all_losses = []
        for t in range(100):  # Train por 100 epochs
            prediction = net(x)     # Input x and predict based on x
            loss = my_loss(prediction, y)     # Must be (1. nn output, 2. target)
            all_losses.append(loss)
            optimizer.zero_grad()   # Clear gradients for next training
            loss.backward()         # Perform backpropagation, compute gradients
            optimizer.step()        # Apply gradients
        # Plotting routine
        plt.plot(all_losses[3:60]) # 10:40
        plt.ylabel("MAPE")
        plt.xlabel("Epoch")
        plt.legend(["lr = 0.01", "lr = 0.005", "lr = 0.0025", "lr = 0.001"])
    return
      