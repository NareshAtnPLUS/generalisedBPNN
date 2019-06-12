from neural_net import BackPropNN
import numpy as np

if __name__ == "__main__":
    inp,out = map(int,input("Enter the input and output neuron dimensions: ").split())
    hid_layers = [int(x) for x in input('Enter the hidden neurons dimensions: ').split()]
    inp = np.random.random((inp))
    print(inp.shape,out,hid_layers,sep="\n")
    bpnn = BackPropNN(inp,hid_layers,out)
    bpnn.neural_net()
