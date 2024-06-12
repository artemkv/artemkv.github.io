## TL;DR

- When in doubt, chain rule
- Sufficiently deep NN can approximate any function
- Model `p(X1, X2,..., Xn)` using conditional independence assumption and a chain rule
- Approximate distributions in the chain using ML algorithms
- Improvise to get different algorithms


## Intro

- Computer graphics: generate an image from a description
- Generating description from graphics is exact inverse of this process
- **Statistical Generative Models** is a probability distribution `p(X)`, based on combination of data (e.g. images of cats) and prior knowledge (e.g. physics, materials)
- Loss function (e.g. maximum likelihood), optimization algorithm are also a part of a prior knowledge
- There is a spectrum, but this course is mostly about data-driven models
- The model is, basically, image `x` → `p(X=x)` → scalar probability `P(x)`
- Sampling from `p(X)` generates new images
- So this model is a data simulator, can generate new data; ideally, controlled (i.e. accepts the prompt in form of text description or a sketch)
- This model also allows you to check how likely it is that the data was generated using this model
- Example of inverse problems: `P(high res|low res)`, `P(color image|greyscale)`, `P(full image|masked image)`, `P(english text|chinese text)`, `P(actions|past observations)`


## Probabilistic background and model assumptions

- Given set of examples (e.g. images of cats) `x_i`, `i=1,2,...,n`, sampled from `Pdata`: `x_i ~Pdata`
- `Pdata` is unknown, and the goal is to get a good approximation for this model
- Because if we have a good approximation of `Pdata`, we could sample from it, and have good samples (e.g. images that look like cats)
- You could also take a new image of cat and ask the model how likely that image was generated using that model
- To find an approximation, you need to come up with family of distributions `Ptheta`, and find the distribution in that family that is closest to `Pdata`
- To do that, you need to be able to measure the distance `d(Pdata, Ptheta)`
- Once you know how to measure the distance, it becomes an optimization problem
- Unfortunately, finding a good distribution family and a notion of a distance is hard, and not clear how to do
- Different families of generative models make different choices
- By learning `Pdata` you end up learning what's common between different images (allows recovering features)
- How to represent `p(X)`? Considering that `x` is an image or text, which means high-dimensional vector
- If every pixel in an image is a random variable, you could generate an image sampling from the joint distribution `p(X1, X2,..., Xn)`
- Assuming black and white images, to specify this distribution you need `2^n-1` parameters
- And once you start considering RGB and larger images, you very quickly run out of atoms in the Universe
- Is there way out?
- You could make an assumption that your variables are independent, which means `p(X1, X2,..., Xn) = p(X1)*p(X2)*...*p(Xn)`
- You would still have the same potential number of images, but you have much fewer parameters (for black and white image, `n` parameters instead of `2^n-1`)
- However, independence assumption is too strong. Such a model would likely not be very useful (imagine generating every pixel of an image independently)
- What about chain rule (always true): `p(X1 & X2 & ... & Xn) = p(X1)*p(X2|X1)*p(X3|X1,X2)*...*p(Xn|X1,X2,...,Xn-1)`
- This does not really help, as you would still need `1+2+...+2^(n-1) = 2^n-1` parameters (same black and white image), but it looks like a step in a right direction
- You could now make a conditional independence assumption: `Xi+1 ⊥ X1,...Xi-1|Xi`, given `Xi`, `Xi+1` is independent of `X1,...Xi-1` (basically, Markov assumption)
- This gives `p(X1, X2,..., Xn) = p(X1)*p(X2|X1)*p(X3|X2)*...*p(Xn|Xn-1)`, which is `2n-1` parameter
- A single word/pixel is probably not enough for a good model though, so maybe this assumption is still a bit too strong
- So instead we can use a Bayesian network approach, where each variable is conditionally dependent on a subset of random variables
- Bayesian network is a DAG with one node for each random variable, and you specify the variable's probability conditioned on its parents' values
- Since you are only conditioning on parents, and as long as there not too many parents, you get big savings in number of parameters
- We will not use Bayesian network directly (that is the subject of CS 228: Probabilistic Graphical Models), instead we will use NNs to represent this model (see below)
- Making this kind of assumptions is similar to Naive Bayes where we make an assumption of words being conditionally independent given the label ("spam"/"not spam"), i.e. given you know whether the mail is spam or not, probability of seeing a word `A` does not tell you anything about probability of seeing word `B`
- Since in reality, words are dependent, this model is not very true, but it is still quite useful
- Once you formulated this kind of model, you estimate your parameters from data
- And then you can reverse the conditions (using the Bayes rule) and predict the label
- Neural models replace conditional distributions (e.g. `p(Xn|X1,X2,...,Xn-1)`) in the chain rule with NNs (you would need a different NN for each position, but still a win)
- And sufficiently deep NN can approximate any function (magic of deep learning), so this is another valid approach to model `p(X)`


### Generative vs discriminative models

- `p(Y,X) = p(X|Y)p(Y) = p(Y|X)p(X)`, this corresponds to 2 Bayesian networks
- `Y → X` is **generative** network (can generate data `X` for a given label `Y`)
- `X → Y` is **discriminative** network (can discriminate label `Y` when given `X`)
- If all you care is `p(Y|X)`, don't bother modeling `p(X)`, estimate `p(Y|X)` from data, and you are done (this is what most of the ML algorithms do, meaning most of them are discriminative)
- Of course, if you have multiple features, then `p(Y|X)` becomes `p(Y|X1,X2...Xn)`, which is also a lot of parameters, but we can always make an assumption `p(Y=1|X, alpha) = f(X, alpha)`, basically replacing the full probability distribution with a function
- Note that this does not require independence assumption!
- We want the function `f(X, alpha)` to output a value between 0 and 1
- We also want `Y` to depend, in some reasonable way, on `X1,X2...Xn`
- We can assume this dependency is specified by a vector `alpha` of `n+1` parameters (linear dependence)
- This is, basically, a recipe for the logistic regression
- And if we use NN, we can model this function using non-linear dependence, which makes model even more powerful
- This is a building block for generative models as well, since the easiest way to build a generative algorithm is to predict `p(X|X-1)` using NN


## Autoregressive models

- Paper: [The Neural Autoregressive Distribution Estimator](https://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)
- Models: FVSBN, NADE, MADE
- This is the algorithm behind large language models (LLMs), e.g. Chat GPT
- **Given:** a dataset `D` of handwritten digits (binarized MNIST, each image is 28x28=784 pixels)
- **Goal:** learn a probability distribution `p(X) = p(X1,X2,...,X784)` such that when you draw `x ~ p(X)`, `x` looks like a digit
- Two-step process:
- 1. Parametrize a model family `Ptheta(X)`
- 2. Search for model parameters `theta` based on training data
- To express `p(X)` using the chain rule, you need to pick an ordering, e.g. raster-scan ordering (from top-left to bottom-right)
- Chain rule: `p(X1,X2,...,X784) = p(X1)*p(X2|X1)*p(X3|X1,X2)*...*p(X784|X1,X2,...,X783)`
- As this is too complex, we need to make some modeling assumptions
- Assume `p(X1,X2,...,Xn) = p(X1;alpha1)*p_logit(X2|X1;alpha2)*p_logit(X3|X1,X2;alpha3)*...*p_logit(Xn|X1,X2,...,Xn-1;alpha_n)`
- Basically, use logistic regression to approximate all the distributions except `p(X1;alpha1)`, which is simple enough to be modeled exactly (in this case it's Bernoulli)
- You had 1 problem, now you have `n-1` problems (every `p_logit` is a separate model with separate parameters)
- And you also need `n^2` parameters in `alpha` vectors (manageable)
- Actually you have a full freedom how to model each of the distributions, logistic regression is just one of the possibilities
- But we've just built an **autoregressive model**
- To evaluate the model, you compute every `p_logit` and multiply
- To sample from `p(X1,X2,...,X784)`, you can sample from every `p_logit` in order
- Unfortunately, the results are not great (it works, but it's shit), because logistic regression is not very good
- So let's replace the logistic regression with a single layer NN
- Having separate NN for each pixel is annoying, could we make it simpler?
- What can be done is to re-use weights `w1` from `p_nn(X2|X1;alpha2)` in `p_nn(X3|X1,X2;alpha3)`, then reuse `w1` and `w2` from `p_nn(X3|X1,X2;alpha3)` in `p_nn(X4|X1,X2,X3;alpha4)` and so on (you could also re-use bias vectors)
- Doing so requires `n` weight vectors
- This makes the model simpler, faster and less prone to overfitting, so it's a big win
- In practice, the results look significantly better too
- If you want to model a continuous distribution, one approach is to learn a uniform mixture of `K` Gaussians (the output of the model is `K` means and `K` sigmas)
- If you need to ensure the output is positive, you could use exponential distributions instead

### Relation to other architectures

- Autoregressive model look a bit similar to autoencoders, but autoencoders don't have any concept of ordering (which is necessary to model the chain rule), all the inputs affect all the outputs
- You could enforce the ordering by masking out some network connections: this is what MADE does
- In MADE, when generating the first pixel, it does not depend on any inputs, for the second pixel, it only depends on the first one, etc.
- Another approach is to use RNN. You still need to pick an ordering
- RNNs are slow and hard to train due to vanishing gradients; but they can produce impressive results
- RNNs also have the constant number of parameters regardless of the sequence length and are very efficient at inference time
- State of art RNNs use attention (2023)
- Current state of the art (GPTs): replace RNN with transformer (transformers are not covered here)
- For images, the natural approach would be to use a CNN. You can do that, however, if you want to adhere to the autoregressive model, you need to make sure that, when you are making prediction, the pixels that you are using for prediction are consistent with the ordering you have chosen (don't look at the future pixels, but you can look at any of the past ones)
- The way to achieve it is, again, by applying masking (to convolutions, meaning `[[X, X, X][X, 0, 0][0, 0, 0]]` where X can be 1 or 0)
- This approach will ignore pixels that are above, but further to the right than the mask can cover (blind spots)
- _My note: personally, I don't see this as a problem in the sense that considering pixel super far to the right to be in the past, just because it is 1 pixel up, is not conceptually very solid. But I do consider a problem not being able to use a large portion of an image for inference_
_My note: the mix of very strict mathematics on one side and completely unjustified heuristics on the other, is quite disturbing_

### Maximum likelihood learning

- Assume all the data is coming from some distribution `Pdata`
- Dataset `D` of `m` samples from `Pdata`
- Assume data points are i.i.d.
- Ideally, we want `Ptheta` to capture `Pdata` precisely, but, in general, this is not achievable
- So we all we can do is to find a "good" approximation
- Definition of "good" depends on the task
- We estimate the similarity based on distance `d(Pdata, Ptheta)`
- One way to estimate the distance between 2 distribution is KL divergence
- And now just look up "Maximum likelihood estimation (MLE)" in the notes on "Statistics, theoretical foundations"
- Because KL divergence leads to MLE
- Your maximization objective is then a `sum [log(Ptheta(X))] over all x`
- `theta` is all possible parameters of a NN
- But evaluating `Ptheta(X=x)` is actually easy: just go and apply the chain rule, iteratively (gets you the probability of 1 image) then multiply across the whole dataset
- How to train, standard ML procedure:
- 1). Initialize all weights at random
- 2). Compute the gradient of the log likelihood, using back prop
- 3). `theta(t+1) = theta(t) + alpha*grad`
- You can do it in minibatches, of course
- As usual, you are looking for the best bias-variance tradeoff (model that is good enough to be close to true distribution, but not so good that it will overfit)








cont with lecture 3, 42:50