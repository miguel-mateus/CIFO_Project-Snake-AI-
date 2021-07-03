import numpy as np
from NN_Parameters import *

def flatener(offspring1): #Flattens the list of arrays of the weigths and biases of the network into a single array
    arr=np.ndarray.flatten(offspring1[0])
    for i in range(len(offspring1)-1):
        
        temp1=np.ndarray.flatten(offspring1[i+1])
        
        arr=np.concatenate((arr,temp1))
    return arr

def reshaper(arr): #Reshapes the array of the weigths of the network back into its original form, so that they can be used in the network
    list1=[]
    ar1=np.reshape(arr[:hidden_size * (window_size**2 + 1)], (hidden_size, window_size**2 + 1)) #Plus one because of bias
    list1.append(ar1)
    ar2=np.reshape(arr[hidden_size * (window_size**2 + 1) : (hidden_size * (window_size**2 + 1)) + (hidden_size * (hidden_size + 1))], (hidden_size, hidden_size + 1))
    list1.append(ar2)
    ar3=np.reshape(arr[(hidden_size * (window_size**2 + 1)) + (hidden_size * (hidden_size + 1)) : (hidden_size * (window_size**2 + 1)) + (hidden_size * (hidden_size + 1)) + (4 * (hidden_size + 1))], (4,hidden_size + 1))
    list1.append(ar3)
    return list1
    