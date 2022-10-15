import numpy as np
from tensorflow import keras
from collections import deque


def Sigmoid(x):
    return 1/(1+ np.exp(-x))

def CostFunction(predicted, expected):
    return np.sum(np.square(np.subtract(predicted, expected)))

def dSigmoid(x):
    return Sigmoid(x) * (1-(Sigmoid(x)))

def dCostFunction(expected, calc):
    return np.array(2 * (expected - calc))

def scale_rows(vector_a, vector_b):
    vector_c = np.zeros(vector_b.shape)
    for entry in len(vector_a):
        vector_c[entry] = vector_a[entry] * vector_b[entry]
    return vector_c
        
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = np.array(train_images)
training_x = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2])

test_images =np.array(test_images)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2])

print(train_labels)
train_y = []
for i in train_labels:
    a = np.zeros(10)
    a[i-1] = 1.0
    train_y.append(a)
    


# for element in a:
#     print(element)

class connection:
    def __init__(self, a, b, w):
        self.a = a
        self.b = b
        self.w = w
    
    def transfer_activation(self):
        return self.a.getActivation() * self.w
    
        
        

class neuron:
    def __init__(self, connections=[]):
        self._Activation = 0
        self.z = -1
        self._connectionsTo = []
        self._connectionsFrom = []
        
    def setActivation(self, actv):
        self._Activation = actv
        
    def getActivation(self):
        return self._Activation
    
    def connectionsFrom(self):
        return self._connectionsFrom
    
    def connectionsTo(self):
        return self._connectionsTo    
    
    def createConnection(self, other:'neuron', weight_scale=5):
        weight = weight_scale* np.random.randn()
        self._connectionsTo.append(connection(self, other, weight))
        other._connectionsFrom.append(connection(self, other, weight))
        
            
        
class layer:
    def __init__(self, neurons):
        self._cntNeurons = neurons
        self._neurons = [neuron() for i in range(self._cntNeurons)] # Creates empty neuron Nodes
        self._bias = [np.random.randint(-2,3) for i in range(self._cntNeurons)]
    def ConnectTo(self, other:'layer'):
        for neuronOt in other._neurons: #other is the layer getting connected
            for neuronSlf in self._neurons: # self is the previous layer
                neuronSlf.createConnection(neuronOt)
    def size(self):
        return self._cntNeurons
    def Neurons(self):
        return self._neurons
    def pass_through(self):
        for i, n in enumerate(self._neurons):
                    activation_sum = 0
                    for connections in n.connectionsFrom(): # Acumulate the activation sum
                        activation_sum+= connections.transfer_activation()
                    n.z = (activation_sum - self._bias[i])# Applies Bias, 
                    n.setActivation(Sigmoid(n.z)) # passes that value to the sigmoid, and then sets that neurons activation

    def getActivations(self):
        return np.array([n.getActivation() for n in self._neurons]) 
    
    def getWeights(self): # Neuron in the Layer x neruon in the previous layer
        weights = []
        for n in self._neurons:
            weights.append(np.array([n.connectionsFrom()[i].w for i in range(len(n.connectionsFrom()))]))
        return np.array(weights)
    
    def setWeights(self, weights):
        i =0
        for n in self._neurons:
            j = 0
            for con in n.connectionsFrom():
                # print(i,j)
                con.w = weights[i, j]
                j+=1
            i+=1
    def getZ(self):
        return np.array([n.z for n in self._neurons])
    
                
        
    

class net:
    
    def __init__(self, layers=()):
        self.layers = []
        previous_layer = None
        for layerSize in layers:
            new_layer = layer(layerSize)
            self.layers.append(new_layer)
            if previous_layer != None:
                previous_layer.ConnectTo(new_layer)
            previous_layer = new_layer
            
    def _PassThrough(self, activations):
        if len(activations) != self.layers[0].size():
            print("CRITICAL ERROR")
            raise Exception("Shape Error...")
        
        #start this neurons Activations
        for i, neurons in enumerate(self.layers[0].Neurons()):
            neurons.setActivation(activations[i])
        
        #Our network needs a Forward Propagation
        for layer in self.layers[1::]:
            layer.pass_through()
    
    def _getFinalActivations(self):
        return self.layers[-1].getActivations()
    
    def predict(self, activations):
        self._PassThrough(activations)
        return self._getFinalActivations()
    
    def back_propagate(self, layer_index, dError:'list'):
        if layer_index == 0:
            return dError[::-1]
        wT = np.transpose(self.layers[layer_index+1].getWeights())
        # print(wT.T.shape)
        delta = np.dot(dError[-1], wT.T)
        delta *= dSigmoid(self.layers[layer_index].getZ())
        dError.append(delta) 
       
        return self.back_propagate(layer_index-1, dError)
        
        
    def train(self, train_data=[], train_labels=[], test_data=[], test_labels=[], learning_rate=0.05, batch_size=5, epochs=1):
        trainCostFunction = []
        accuracy = 1.0
        predictions = []
        for i in range(len(train_data)): #i: Index of current training set
            layer_index = len(self.layers)-1 #layer index will
            prediction = self.predict(train_data[i])
            predictions.append(np.argmax(prediction) == np.argmax(train_labels[i]))
            trainCostFunction.append(CostFunction(train_labels[i], prediction))
            
            dc = dCostFunction(prediction,train_labels[i])
            
            ## Propogate the error backwords and get a delta matrix
            delta_matrix = self.back_propagate(layer_index-1, [dc * dSigmoid(self.layers[-1].getZ())])
            
            for lIndex, iLayer in enumerate(self.layers):
                if lIndex == 0:
                    continue
                iLayer.setWeights (
                    iLayer.getWeights() - learning_rate * np.dot(np.transpose(iLayer.getActivations()), delta_matrix[lIndex])
                )
            if i % 25 == 0:
                print('Cost Function Update:', trainCostFunction[-1])
                accuracy = sum(predictions) / len(predictions)
                print('Accuracy:', accuracy)
        return trainCostFunction
            
            
            
    

my_net = net((784,5, 5, 10))

my_net.train(training_x, train_y)
    
        
        
# Layer1(784 Neurons) -> Layer2(64 Neurons) -> Layer3(32 Neurons) -> Layer4(10 Neruons)