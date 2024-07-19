# Deep Learning
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Stanford CS 230: Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)

## TL;DR

- Use ReLU, can try tanh, never sigmoid
- Initialize weighs as `np.random.randn(n) * sqrt(2.0/n)`
- Use batch normalization
- Use softmax + cross-entropy loss for classification, L2 squared norm for regression
- Use L2 regularization
- Use Adam


## Neural networks - Biological motivation

- Fun fact: in 2010 MIT almost kicked NNs out of its course on AI, as they didn't look any good
- **Neural networks** are inspired by the goal of modeling biological neural systems
- The artificial neural network is made of artificial neurons organized in layers
- The biological brain, specifically human cerebral cortex is made up of 6 layers
- Every neuron receives an input signal and produces the output signal
- Input layer receives the raw data from a sensor (like an eye)
- Output layer neurons provide the final answer (is it a cat or a dog)
- Hidden layers of neurons are responsible for extracting features (edges and contours)
- Hidden layers have much fewer neurons than the input layer, forcing the network to use compressed representation of the original input. This leads to generalization
- Input signal of a hidden neuron is an output from neurons at the previous layer
- In a fully-connected network, every neuron has its output connected to the input of every neuron in the next layer
- Neurons from the previous layer do not contribute to the input equally, the input from each previous neuron is multiplied by a specific weight
- The weighted input summed together produces the **logit** of the neuron
- In many cases, the logit also includes a bias, which is a constant
- The logit is then passed through an activation function, e.g. "sigmoid"
- In real neuron, the signal only passes the synapse if it reaches a certain threshold
- Bias plays a role of a threshold for an activation function. Without bias, you would need to use more complicated mathematical model
- In a feed-forward network, connections only traverse from a lower layer to a higher layer
- Number of layers and number of neurons in the layer is a choice you have to make
- In practice, 3-layer neural networks (2 hidden layers + 1 output layer) will often outperform 2-layer nets, but going even deeper (4,5,6-layer) rarely helps much more
- This model of a biological neuron is very coarse. Comparing it to the real biological brain is misleading
- The whole area of research has now diverged from modeling biological brain to solving practical engineering tasks using ML approach


## Neural networks - Mathematical interpretation

- Instead of appealing to brain analogies, we can think about neural network as a function, computing output `y` for an input `x`
- We use this function as a model for predicting values `y` for unknown values of `x`
- For example, 2-layer network can be mathematically expressed as `y = W2*g(W1*X)`, where `g` is some activation function
- An activation function is required add some non-linearity to the model
- Without an activation function, the two matrices `W1` and `W2` would collapse to a single matrix `W3` and the model would become a linear classifier: `y = W3 * X`
- There are multiple activation function to choose from. Some well-known activation functions are **Sigmoid**, **Tanh** and **ReLU** (see below)
- You can think about neural network as a function approximator (for some "true" function)


## Classification and Softmax layer

- The most trivial application of neural networks is classification
- Label is the value that we attribute to the certain input when doing classification (a category)
- Softmax layer is a special kind of the output layer where output of each neuron represents the probability of the input being attributed one of mutually exclusive labels
- The sum of all the outputs of softmax layer should be equal to 1


## Optimization

- **Error** is the difference between predictions and actual observations
- **Loss function** (sometimes also called "cost function") measures the quality of a particular set of parameters of a neural network
- Loss function can be expressed through error (e.g. **mean square error**)
- You choose a loss function that better suits the task you want to perform (see below)
- The output of a given neural network depends only on the input (training data) and weights
- Training data is given and fixed; the weights as variables we have control over
- **Optimization** is the process of finding the set of weights that minimize the loss function
- Basically, finding the set of weights that, applied to the inputs produce the outputs that are the closest to expected
- It is not practically possible to find the correct weights directly by solving the system of equations
- Instead, we are going to start with a random set of weights and find the correct weights through the series of small adjustments (iterative refinement)
- At any particular point, the partial derivative of a loss function by any weight `i,j` is a measure of impact of that weight on the loss function
- The gradient is a vector of partial derivatives
- The gradient points in the direction of the greatest rate of increase of the function, and its magnitude is the slope of the graph in that direction
- So we are going to move in a direction opposite to the gradient
- The differentiation give us the expression for the gradient, in terms of inputs, outputs and weights, given the specific activation function, then we have to evaluate this expression by plugging in specific values of inputs, outputs and weights, to find the direction at that specific point
- **Gradient descent** is the procedure of repeatedly evaluating the gradient and performing a weight update
- Expressing the loss function in terms of weights involves multiple composed functions, so taking the partial derivative of a loss function directly (using Calculus) can still be very challenging (and error-prone)
- For the outer layer weights it may be quite easy, but it gets more and more complicated as we go towards the input layer. This is because dependency of error on the inner layer inputs is very indirect
- But what if we could somehow know the error at each layer? We could then do the same calculation layer by layer until we get to the input layer. This is what backpropagation does
- **Backpropagation** is a way of computing gradients of expressions through recursive application of **chain rule**
- Chain rule allows computing gradients of any arbitrary complex expressions. You represent any expression as a computational graph, calculate the local gradient at every node, multiply it by an upstream gradient and pass it downstream
- Understanding of this process and its subtleties is critical to understand, and effectively develop, design and debug neural networks
- In a nutshell, the gradient that goes to any weight is a product of the input of that neuron and the upstream gradient
- The gradient tells us the direction in which the function has the steepest rate of increase, but it does not tell us how far along this direction we should step
- This is why we moderate the adjustments by multiplying the gradient by the **step size**, also called the **learning rate**
- Learning rate is one of the most important hyperparameters in training a neural network
- The small value of the learning rate can slow down training, especially in case of flat error functions
- The big value of the learning rate can result in overshooting the minimum and effectively nullifying the improvements made during earlier iterations
- One possible solution is to vary the learning rate so that the closer we are to the minimum, the smaller are the steps
- With **batch gradient descent** you calculate the loss, gradient and weight adjustments from the whole dataset
- This allows you to do all the calculations in a matrix form, which is very efficient
- However, you need to go through the whole dataset before you can adjust weights, and you need the dataset to fit into the memory, which can become very challenging when working with huge datasets
- When you apply gradient descent to adjust weights for every single sample, it is called **stochastic gradient descent**
- Stochastic gradient descent is not computationally efficient, so it's not very common in practice
- Weights can also be adjusted in minibatches, where you compute the gradient over small batches of the training data


## Optimizers

- The commonly supported idea used to be that learning process may get stuck in local minima instead of arriving at the actual minima
- Researches have proven that this is not the case, local minima probability in the `n`-dimentional space is extremely small
- Instead, the biggest challenge is the presence of **saddle points**, where surface is either flat or the point is a minimum in one dimension and maximum in another (think horse saddle)
- So the biggest difficulty is to determine the right direction to go, but eventually you should get to global minima
- Another problem appears when the loss changes quickly in one direction and slowly in another (taco shell). In that case SGD will start zigzagging, resulting in a very slow learning
- There are various algorithms that try to tackle these problems, called **optimizers**
- For example, you might use the momentum to decide on direction, so that you decide not only based on where you are, but also on where you were going at the previous step. With momentum, you are gradually picking up speed
- **Nesterov Momentum** is a slightly different version of the momentum update and works slightly better than standard momentum
- With Nesterov Momentum instead of evaluating gradient at the current position, we know where our momentum is about to carry us, and we evaluate the gradient at this "looked-ahead" position
- The downside of using momentum is that it usually picks up so much speed that it overshoots the minima, and then has to come back. This however, can be a good thing, since it helps to jump out of the local minimum or a very narrow minimum that exists because of the noise
- You can also consider decreasing the learning rate progressively, using step decay (drops the learning rate by a factor every few epochs) or exponential decay
- Much work has gone into devising methods that can adaptively tune the learning rates
- An adaptive learning rate removes the need to choose the learning rate manually and will generally outperform a model with a badly configured learning rate
- **Adagrad** is an adaptive learning rate method. Using Adagrad, the weights that receive high gradients will have their effective learning rate reduced, while weights that receive small or infrequent updates will have their effective learning rate increased. This helps with taco shell scenario
- A downside of Adagrad is that the monotonic decrease in learning rate usually proves too aggressive and stops learning too early. Because of that it's not widely used in practice
- **Adadelta** is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done
- **RMSProp** is another attempt to adjust Adagrad method to reduce its aggressive, monotonically decreasing learning rate. RMSprop is a very effective in practice, but it is currently unpublished
- **Adam** is a recently proposed algorithm that is "kind of like RMSProp with momentum". It is straightforward to implement, computationally efficient, has little memory requirements and works very well in practice
- TLDR: Adam is currently recommended as the default algorithm to use
- As an alternative, it is worth trying SGD + Nesterov Momentum. Then, if this approach is working, you can try to improve it further by using decay
- In any case, it has been proven that smarter network architectures that are easier to train for the specific task (e.g. CNN) are much more important than optimizing walking the error surface


## Activation functions

TL;DR
- Use ReLU
- Never use sigmoid
- Try tanh, but expect it to work worse than ReLU

### Sigmoid

- **Sigmoid** is `1/(1 + exp(-x))`
- Sigmoid used to be a common choice in the past, but has fallen out of favor and is rarely used today because of its major drawbacks
- Problem 1: Sigmoids saturate and kill gradients
- On large positive or negative inputs sigmoid has a very flat activation curve. We say that the neuron's activation **saturates** at either tail of 0 or 1
- The gradient at these regions is almost zero. This will effectively "kill" incoming upstream gradient and almost no signal will flow through the neuron to its weights and recursively downstream. The network will barely learn
- Problem 2: Sigmoid outputs are not zero-centered
- The output of sigmoid is always positive, so the data coming into a neuron on the next layer is always positive
- As a result, during backpropagation, the gradient on the weights will have the same sign as an upstream gradient, for all weighs
- So the weight adjustments (deltas) will we either all positive or all negative, which can require zigzagging to follow the real gradient
- Problem 3: exp function is somewhat expensive to calculate

### Tanh

- **Tanh** is `(exp(x) - exp(-x))/(exp(x) + exp(-x))`
- In practice the tanh is always preferred to the sigmoid
- Zero-centered, but still kills gradients when saturated

### ReLU

- **ReLU** is `max(0, x)`
- ReLU has become very popular in the last few years
- Is very fast to calculate, derivative is super easy too
- Greatly accelerates the convergence of stochastic gradient descent
- Does not saturate in the positive region
- Still saturated and kills gradients in the negative region
- Up to 40% of ReLU units can irreversibly die during training. Once ReLU input gets below zero, gradient gets killed, and the zeroes start flying in both directions. There is a good chance such ReLU is never going to recover
- In practice it is still performing much better than sigmoid or tahn

### Leaky ReLU

- **Leaky ReLU** is `max(0.01*x, x)`
- This makes negative part of ReLU a negative slope with an angle 0.01, not a flat line
- 0.01 can be made into a parameter learned during backpropagation
- Leaky ReLUs are one attempt to fix the "dying ReLU" problem, but the results are not always consistent


## Loss functions

- Classification: SVM loss, squared hinge loss, softmax + cross-entropy loss
- When the set of labels is very large, it may be helpful to use hierarchical softmax
- Regression: L2 squared norm


## Data preparation

- Neural network do not work well if the input and output is not prepared correctly
- All the techniques common for the supervised learning apply (e.g. normalization)
- **Saturation** is a problem specific to neural networks. It happens when large signals, sometimes driven by large weights, result in a shallow slopes of the activation function, driving gradients to zero (this depends on the choice of activation functions)
- TODO: Inputs need to be scaled to be small, but not zero. Common ranges are `0.01 .. 0.99` and `-0.99 .. 0.99` (HOW TO CHOOSE? SEEM TO BE CONFLICTING IN DIFFERENT SOURCES)
- TODO: CS229 simply suggested normalizing the input by subtracting mu and dividing by sigma. But: what is true sigma? CS229 says we need to use the training set sigma, but in theory, I guess, we need a population sigma. Should we treat training set as a population or as a sample? Am I overthinking it?
- Outputs should be within the range that the activation function can produce, otherwise they will lead to saturation. A good range is `0.01 .. 0.99` (RE-VISIT)


## Weight initialization

- Neural network do not work well if the weights are not initialized correctly
- When every weight is initialized with 0 (or simply the same value), the network has zero chance to learn anything. In that case every neuron computes the same output, the same gradients during backpropagation and undergo the exact same parameter updates
- Choosing large values for weights can lead to saturation and killing of gradients (depending on choice of activation functions). For that reason you might want to initialize weights with the random values that are very close to zero, but not zero
- On the other hand, choosing very small weights can lead to **vanishing gradients** problem. Neural network that has very small weights will compute very small gradients during backpropagation, with gradients getting progressively smaller at every layer
- One recommended heuristic is to initialize the weights of each neuron as `np.random.randn(n) / np.sqrt(1.0/n)`, where `n` is the number of neuron's inputs (Xavier initialization). This is a good place to start
- However, this method is not recommended with ReLU units, as it may lead to dying ReLUs. The current recommendation is to use ReLU units and initialize weighs as `np.random.randn(n) * sqrt(2.0/n)`, where `n` is the number of neuron's inputs
- You should monitor distributions of weights on different layers of the neural network

### Batch normalization

- Initializing the weights boils down to choosing the values that produce "good" distributions of activations (no saturation, good gradients coming back)
- The idea is, instead of tweaking the weights, we could preprocess (normalize) the activations at every layer
- **Batch Normalization** is forcing the activations throughout a network to take on a unit gaussian distribution
- Unit gaussian (normal) distribution is a normal distribution with the mean of 0 and standard deviation of 1
- You can also scale and shift the distribution by some parameters that neural network could learn during training
- Batch normalization is a recently developed technique (2015)
- It makes neural networks significantly more robust to bad initialization, so it has become a very common practice
- In practice, all you need is to insert the `BatchNorm` layer immediately after fully connected or convolutional layers, and before non-linearities
- In case of CNN, batch normalization is applied on every activation map separately


## Test set, validation set, and overfitting

- You should split your dataset into **training data**, **validation data** and **test data**
- Never evaluate the model using the same data you used to train it!
- Very complex model (i.e. very deep neural network) may have enough degrees of freedom to simply adjust itself to fit every sample, but perform very badly on new data
- In that case we end up "memorizing" every sample that we have seen instead of generalizing. This is called **overfitting**
- You should stop the training process as soon as you start overfitting the network. To do this, you should divide the training process into **epochs**
- An **epoch** is a single iteration over the entire training set
- Validation data is used to estimate the training efficiency at the end of each epoch. We only do the next iteration if the accuracy of the prediction on the validation data keeps increasing
- Validation data can also be used to optimize hyperparameters like learning rate
- In practice, you should not be using smaller networks because you are afraid of overfitting
- Instead, you should use as big of a neural network as your computational budget allows, and use regularization techniques that control overfitting
- **Regularization** modifies the error function by bringing the weight values into the error function. This way we are adjusting the weight not only in the way that minimizes the error but also in the way that satisfies certain criteria about the weight itself
- L2 regularization heavily penalizes peaky weight vectors and prefers diffuse weight vectors
- L1 regularization leads the weight vectors to become sparse. Neurons with L1 regularization end up using only a sparse subset of their most important inputs and become nearly invariant to the "noisy" inputs
- In practice, L2 regularization is expected to give superior performance over L1. It is a good place to start
- **Dropout** is another method that works by only keeping a neuron active with some probability while training
- Also, you don't have to train your network from scratch. Instead, you can use already trained network, drop the last layer and train it from that point. This is the idea of **transfer learning**


## Sanity checks

- Look at the loss on the untrained network with regularization strength set to zero
- For a softmax classifier with 10 classes we expect a diffuse probability of 0.1 for each class. Softmax loss is the negative log probability of the correct class `-ln(0.1) = 2.302`. So we would expect the initial loss to be 2.302
- Try increasing the regularization strength, this should increase the loss
- Overfit a tiny subset of data. Before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero loss (set regularization to zero)


## Hyperparameter optimization

- The most important hyperparameter you should find first is learning rate
- Start with small regularization and find learning rate that makes the loss go down
- Usual range for learning rate is between 1e-3 and 1e-05
- For the combinations of hyperparameters, try values from different ranges on small number of epochs (e.g. 5)
- Search for hyperparameters on log scale (e.g. 0.01, 0.001, 0.0001)
- Careful with best values on border. If the final learning rate is at the edge of the interval, you may be missing more optimal hyperparameter setting beyond the interval
- Search for good hyperparameters with random search, not grid search. Randomly chosen trials are more efficient for hyperparameter optimization than trials on a grid
- Re-fine the value by doing longer runs with smaller variations of a hyperparameter
- This way you can discover the best network architecture, learning rate, decay and regularization values
- Lectures on NLU suggest using statistical hypothesis testing based comparison of model performance with different hyperparameters, mention specifically the Wilcoxon signed-rank test. The main idea is to assess the model performance repeatedly, to see that the model that you want to pick is doing better consistently over multiple experiments
- They also mention McNemar's test in the situation when it's too expensive to evaluate a model performance repeatedly


## Training - things to track

- Loss function
- Validation/training accuracy
- Ratio of weights to weight updates
- Activation/gradient distributions per layer
- First-layer visualizations


APPENDIXES

## Tools

- Theano, TensorFlow and PyTorch are mathematical libraries that allow you to describe mathematical functions working on scalars, vectors, matrices and tensors. You are basically writing expressions that operate on variables (mathematical variables, not programming variables)
- TensorFlow is from Google, so it's ugly, but it works on many platforms (e.g. mobile)
- Keras is library on top of TensorFlow (also supports Theano). It is a high-level API to TensorFlow, and allows you to easily build NN without manually programming gradient descent etc.
- PyTorch is like TensorFlow + Keras. It is from Facebook, so it's beautiful. Very good for research
- scikit-learn is a very high level library for machine learning. In scikit-learn you can build NN in just one line. It is plug-and-play API for all classical ML algorithms in the sense that you can simply replace NN with decision tree and keep the rest of the code exactly the same. Personally, I think it's awesome
