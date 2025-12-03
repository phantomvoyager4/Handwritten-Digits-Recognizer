from HDR import data_handling, Layer, Activation, Softmax, Loss, Optimizer_SGD, Backpropagation
pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
images, labels = data_handling(pathimagess, pathlabelss)


hidden_layer1 = Layer(n_inputs=784, n_neurons=128)
activation1 = Activation()
activation2 = Activation()
hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
output_layer = Layer(n_inputs=64, n_neurons=10)
loss_activation = Backpropagation()
optimization = Optimizer_SGD(0.5)

for epoch in range (1000):
    first_layer = hidden_layer1.fpropagation(images)
    first__layer_activation = activation1.forward(first_layer)
    second_layer = hidden_layer2.fpropagation(first__layer_activation)
    second_layer_activation = activation2.forward(second_layer)
    output_layer_output = output_layer.fpropagation(second_layer_activation)
    loss = loss_activation.forward(output_layer_output, labels)