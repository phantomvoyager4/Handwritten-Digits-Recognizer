# HANDWRITTEN DIGITS RECOGNIZER USING NUMPY + IDX2NUMPY (神经网络)

### "Hello world" of neural networks, explained for begginers using math + code.

This is documentation for digit recognizing neural network. I wrote everything without using AI, using only python + numpy, for efficent matrix calculations and idx2numpy to handle data input. <br />
## Table of Contents: 
1. [Data Handling](#1-data-handling)
2. [What is a neuron?](#2-what-is-a-neuron)
3. [Layers](#3-layers)
4. [Activation function](#4-activation-function)
5. [Softmax probability](#5-softmax-probability)
6. [Learning process](#6-learning-process)
7. [Other](#7-other)

## 1. Data handling
[MNIST dataset]("http://yann.lecun.com/exdb/mnist/") has been utilized for network training, due to sample size ($N = 60000$) and efficent data format.<br/>
Dataset stores examples in file `train-images.idx3-ubyte` as a matrix with shape $S = (60000, 28, 28)$, which means there is $60000$ matrixes (examples), each having $28$ rows and $28$ columns. These matrixes store our images, specifically their gray scale in range from $0$ to $255$, where $0 = white$ and  $255 = black$. First of all, we have to reshape our matrix from $(60000, 28, 28)$ to $(60000, 784)$, because we have to hand each gray scale value separately as a single input into each neuron in our network. now, we have a set of $60000$ vectors each having $28 \times 28 = 784$ values.<br />
File `train-labels.idx1-ubyte` is a matrix with shape $S = (60000, 1)$, which stores labels corresponding to each example matrix (e.g. if example image represents digit 7, labels has value $[7]$). Using [idx2numpy](https://pypi.org/project/idx2numpy/) library we transfer these matrixes to numpy array objects, also reshaping our images matrix using numpy function `np.reshape`:
```python
images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
labels = idx2numpy.convert_from_file(pathlabels)
```
Next step is normalization, where we take original examples gray scale values and divide them by maximum value of gray scale, which is $255$. 
```python
images = images / 255.0 
```
Now, our values are represented as float numbers in range from $0$ to $1$ which saves compute power and training time due to simplified calculations. let's visualize how a single example vector looks like as of now:
$$
\mathbf{v} = 
\begin{bmatrix}
0.1 \\ 0.2 \\ 0.5 \\ \vdots \\ 0.3 \\ 0.1 \\ 0.4
\end{bmatrix}
\quad 
\left. \vphantom{\begin{bmatrix} 0.1 \\ 0.2 \\ 0.5 \\ \vdots \\ 0.3 \\ 0.1 \\ 0.4 \end{bmatrix}} \right\} 
\text{784 values,}
\qquad 
\text{where } \mathbf{v} \in \mathbb{R}^{784}
$$
In the next step I will show how are these vectors processed via computer.

## 2. What is a neuron?
Neuron is an atomic part of our network. Instead of visualizing it, we should treat it like a very simple, linear function: $$ N = W \times X + B $$ Where:
$$ \begin{aligned} W &= \text{Neuron weight} \\ X &= \text{Input} \\ B &= \text{Bias} \end{aligned} $$
$W$, neuron weight, is a slope of our function (just like $a$ in $y = ax + b$) 








## 3. Layers


## 4. Activation function


## 5. Softmax probability


## 6. Learning process


## 7. Other


