import numpy as np

class BackPropNN:
    def __init__(self,inp,hidden,out):
        self.inp = inp
        self.weights,self.bias = [],[]
        self.weights.append(np.random.random((inp.shape[0],hidden[0])))
        for i in range(len(hidden)-1):
            self.weights.append(np.random.random((hidden[i],hidden[i+1])))
            self.bias.append(np.random.random((hidden[i],1)))
        self.weights.append(np.random.random((hidden[i+1],out)))
        self.weights = np.array(self.weights)
        print(self.bias.shape)

    def neural_net(self):
        pass