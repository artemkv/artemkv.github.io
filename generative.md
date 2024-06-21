# Deep Generative Models
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

[Stanford CS236: Deep Generative Models](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8)


## TL;DR

- When in doubt, chain rule
- Sufficiently deep NN can approximate any function
- We can think of any kind of observed data `D` as a finite set of samples from an underlying distribution `Pdata`
- The goal of any generative model is then to approximate this data distribution given access to the dataset `D`
- **Autoregressive**: model `p(x1, x2,..., xn)` using conditional independence assumption and a chain rule
- Approximate distributions in the chain using ML algorithms
- **Variational autoencoder**: model `p(x1, x2,..., xn)` using the law of total probability and a latent variable
- Jointly optimize for `p(z)` and `p(x|z)`
- **Normalizing flows**: model `p(x)` as Gaussian noise passed through multiple layers of reversible transformations
- Train by maximizing the likelihood (and minimizing KL divergence between the distribution you produce and distribution of data)
- **GANs**: 2 player minimax game between generator and discriminator. Train by stochastic gradient ascent/descent
- **Energy based models**: turn any function into probability distribution by normalizing by the total volume; don't need to normalize if all you want is to compare
- **Score based models**: replace distribution with score (field of gradients on the distribution), optimize by Fisher divergence instead of maximizing the likelihood


## Intro

- Computer graphics: generate an image from a description
- Generating description from graphics is exact inverse of this process
- **Statistical Generative Model** is a probability distribution `p(x)`, based on combination of data (e.g. images of cats) and prior knowledge (e.g. physics, materials)
- Loss function (e.g. maximum likelihood), optimization algorithm are also a part of a prior knowledge
- There is a spectrum, but this course is mostly about data-driven models
- The model is, basically, image `x` → `p(x)` → scalar probability `P(X=x)`
- Sampling from `p(x)` generates new images
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
- How to represent `p(x)`? Considering that `x` is an image or text, which means high-dimensional vector
- If every pixel in an image is a random variable, you could generate an image sampling from the joint distribution `p(x1, x2,..., xn)`
- Assuming binary black and white images, to specify this distribution you need `2^n-1` parameters
- And once you start considering RGB and larger images, you very quickly run out of atoms in the Universe
- Is there way out?
- You could make an assumption that your variables are independent, which means `p(x1, x2,..., xn) = p(x1)*p(x2)*...*p(xn)`
- You would still have the same potential number of images, but you have much fewer parameters (for black and white image, `n` parameters instead of `2^n-1`)
- However, independence assumption is too strong. Such a model would likely not be very useful (imagine generating every pixel of an image independently)
- What about chain rule (always true): `p(x1 & x2 & ... & xn) = p(x1)*p(x2|x1)*p(x3|x1,x2)*...*p(xn|x1,x2,...,xn-1)`
- This does not really help, as you would still need `1+2+...+2^(n-1) = 2^n-1` parameters (same black and white image), but it looks like a step in a right direction
- You could now make a conditional independence assumption: `Xi+1 ⊥ X1,...Xi-1|Xi`, meaning: given `Xi`, `Xi+1` is independent of `X1,...Xi-1` (basically, Markov assumption)
- This gives `p(x1, x2,..., xn) = p(x1)*p(x2|x1)*p(x3|x2)*...*p(xn|xn-1)`, which is `2n-1` parameter
- A single word/pixel is probably not enough for a good model though, so maybe this assumption is still a bit too strong
- So instead we can use a Bayesian network approach, where each variable is conditionally dependent on a subset of random variables
- Bayesian network is a DAG with one node for each random variable, and you specify the variable's probability conditioned on its parents' values
- Since you are only conditioning on parents, and as long as there not too many parents, you get big savings in number of parameters
- We will not use Bayesian network directly (that is the subject of CS 228: Probabilistic Graphical Models), instead we will use NNs to represent this model (see below)
- Making this kind of assumptions is similar to Naive Bayes where we make an assumption of words being conditionally independent given the label ("spam"/"not spam"), i.e. given you know whether the mail is spam or not, probability of seeing a word `A` does not tell you anything about probability of seeing word `B`
- Since in reality, words are dependent, this model is not very true, but it is still quite useful
- Once you formulated this kind of model, you estimate your parameters from data
- And then you can reverse the conditions (using the Bayes rule) and predict the label
- Neural models replace conditional distributions (e.g. `p(xn|x1,x2,...,xn-1)`) in the chain rule with NNs (you would need a different NN for each position, but still a win)
- And sufficiently deep NN can approximate any function (magic of deep learning), so this is another valid approach to model `p(x)`


### Generative vs discriminative models

- `p(y,x) = p(x|y)p(y) = p(y|x)p(x)`, this corresponds to 2 Bayesian networks
- `Y → X` is **generative** network (can generate data `X` for a given label `Y`)
- `X → Y` is **discriminative** network (can discriminate label `Y` when given `X`)
- If all you care is `p(y|x)`, don't bother modeling `p(x)`, estimate `p(y|x)` from data, and you are done (this is what most of the ML algorithms do, meaning most of them are discriminative)
- Of course, if you have multiple features, then `p(y|x)` becomes `p(y|x1,x2...xn)`, which is also a lot of parameters, but we can always make an assumption `p(Y=1|x, alpha) = f(x, alpha)`, basically replacing the full probability distribution with a function
- Note that this does not require independence assumption!
- We want the function `f(x, alpha)` to output a value between 0 and 1
- We also want `Y` to depend, in some reasonable way, on `X1,X2...Xn`
- We can assume this dependency is specified by a vector `alpha` of `n+1` parameters (linear dependence)
- This is, basically, a recipe for the logistic regression
- And if we use NN, we can model this function using non-linear dependence, which makes model even more powerful
- This is a building block for generative models as well, since the easiest way to build a generative algorithm is to predict `p(x_i|x_i-1)` using NN


## Autoregressive models

- Paper: [The Neural Autoregressive Distribution Estimator](https://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)
- Models: FVSBN, NADE, MADE
- This is the algorithm behind large language models (LLMs), e.g. Chat GPT
- **Given:** a dataset `D` of handwritten digits (binarized MNIST, each image is 28x28=784 pixels)
- **Goal:** learn a probability distribution `p(x) = p(x1,x2,...,x784)` such that when you draw `x ~p(x)`, `x` looks like a digit
- Two-step process:
- 1. Parametrize a model family `Ptheta(x)`
- 2. Search for model parameters `theta` based on training data
- To express `p(x)` using the chain rule, you need to pick an ordering, e.g. raster-scan ordering (from top-left to bottom-right)
- Chain rule: `p(x1,x2,...,x784) = p(x1)*p(x2|x1)*p(x3|x1,x2)*...*p(x784|x1,x2,...,x783)`
- As this is too complex, we need to make some modeling assumptions
- Assume `p(x784|x1,x2,...,x783)` is simply `Bernoulli(f(x1,x2,...,x783))`, where parameter of that Bernoulli is a function of all preceding random variables `X1,X2,...,X783` (in case of binary black and white image)
- This can be modeled using `p(x1,x2,...,xn) = p(x1;alpha1)*p_logit(x2|x1;alpha2)*p_logit(x3|x1,x2;alpha3)*...*p_logit(xn|x1,x2,...,xn-1;alpha_n)`
- Basically, use logistic regression to approximate all the distributions except `p(x1;alpha1)`, which is simple enough to be modeled exactly (it's Bernoulli)
- You had 1 problem, now you have `n-1` problems (every `p_logit` is a separate model with separate parameters)
- And you also need `n^2` parameters in `alpha` vectors (manageable)
- Actually you have a full freedom how to model each of the distributions, logistic regression is just one of the possibilities
- But we've just built an **autoregressive model**
- To evaluate the model, you compute every `p_logit` and multiply
- To sample from `p(x1,x2,...,x784)`, you can sample from every `p_logit` in order
- Unfortunately, the results are not great (it works, but it's shit), because logistic regression is not very good
- So let's replace the logistic regression with a single layer NN
- Having separate NN for each pixel is annoying, could we make it simpler?
- What can be done is to re-use weights `w1` from `p_nn(x2|x1;alpha2)` in `p_nn(x3|x1,x2;alpha3)`, then reuse `w1` and `w2` from `p_nn(x3|x1,x2;alpha3)` in `p_nn(x4|x1,x2,x3;alpha4)` and so on (you could also re-use bias vectors)
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
- Your maximization objective is then a `sum [log(Ptheta(x))] over all x`
- `theta` is all possible parameters of a NN
- But evaluating `Ptheta(X=x)` is actually easy: just go and apply the chain rule, iteratively (gets you the probability of 1 image) then multiply across the whole dataset
- How to train, standard ML procedure:
- 1). Initialize all weights at random
- 2). Compute the gradient of the log likelihood, using back prop
- 3). `theta(t+1) = theta(t) + alpha*grad`
- You can do it in minibatches, of course
- As usual, you are looking for the best bias-variance tradeoff (model that is good enough to be close to true distribution, but not so good that it will overfit, bias and variance in ML terms)
- Note: there is a technicality that involves switching from log-likelihood to empirical log-likelihood. Because the "all x" in the optimization objective actually means "all possible `x` that `Ptheta(x)` is defined on", but all we have is data, so we will use "all observed `x`". This estimator has no bias, and the variance gets reduced by increasing the number of samples (here we mean bias and variance in statistical terms)
- Autoregressive models are good, but you have to pick an ordering, generation is sequential and slow, and you cannot learn features in an unsupervised fashion


## Variational Autoencoders (VAEs)

- Reminder: `P(X=x) = sum [P(X=x|Z=z)*P(Z=z)] over all z = sum [P(X=x,Z=z)] over all z` (The law of total probability, then just expressing joint probability from conditional one)
- As we have seen, modeling `p(x)` directly can be a very complex task, and the distribution may be extremely complex
- But this is because we don't know anything about the underlying structure or meaning of `X`
- If we knew that `X` depended on some variable `Z`, we might have found that modeling `p(x|z)` is significantly easier, and may even produce some easy distributions
- As an example, if you look at many pictures of faces and all you see is pixels, there is going to be a lot of variability, due to gender, age, hair color, etc.
- But if you looked only at images of young blond females, there would be much less variability
- Unfortunately, unless images are annotated, these factors of variation are not explicitly available (latent)
- However, we can explicitly model these factors using latent variables `Z1...Zn`
- This corresponds to `Z → X` bayesian network, where `Z`s are latent high level features (e.g. eye color)
- If you choose `Z`s well, modeling `p(x|z)` can be much easier than modeling `p(x)`
- And if we trained such a model, we could then infer the latent variables, i.e. we could get `p(z|x)` (e.g. `p(EyeColor = blue|x)`)
- The easiest example ever is the mixture of Gaussians: a net `Z → X` where `Z` is categorical, and `p(x|Z=k) = Gaussian(mu_k, sigma_k)`, and you have a table with values of `mu` and `sigma` for each category `k`
- In reality, however, the number of categories usually would be unknown, so you can treat it as a hyperparameter
- And instead of specifying the latent variables by hand, we are going to let our model figure it out
- Our model will rely on 3 key assumptions
- First, we will assume some distributions `p(z)` for our latent variable(s) `Z` (aka priors), it's convenient to pick something easy e.g. `p(z) = Gaussian(0,1)` (but can be any distribution)
- Second, we will assume some distribution `p_theta(x|z)`, and again, it's convenient to pick something easy e.g. Gaussian (but can be any distribution)
- Finally, we will assume that parameters `theta` of `p_theta(x|z)` depend on `Z` in some none-linear way, e.g. `p_theta(x|z) = Gaussian(mu_theta(z), sigma_theta(z))`
- So while keeping all the distributions simple, all the complexity will go into transformations `mu_theta` and `sigma_theta` that can be very complex functions, and we will approximate them using NNs
- Our hope is by fitting this model, it will discover some useful latent features (basically, clusters of data with very low variability)

### VAE optimization

- A very good explanation can be found here: [CS 285: Lecture 18, Variational Inference, Part 1](https://www.youtube.com/watch?v=UTMpM4orS30), [CS 285: Lecture 18, Variational Inference, Part 2](https://www.youtube.com/watch?v=VWb0ZywWpqc), [CS 285: Lecture 18, Variational Inference, Part 3](https://www.youtube.com/watch?v=4LuA5m5Hsxc), [CS 285: Lecture 18, Variational Inference, Part 4](https://www.youtube.com/watch?v=_W2eVLi8rQA)
- The problems begin when you want to evaluate `P(X=x)`, since you need to integrate `p(x, z; theta) over all z`, and this can be a nasty integral to calculate
- This can quickly become intractable even in case of discrete `Z`: suppose we have 20 binary latent variables, evaluating `P(X=x)` involves a sum with 2^20 terms
- And to fit this model, you actually need to evaluate `P(X=x)` for all the datapoints in the dataset
- Ways to cheat: Monte-Carlo. Instead of summing/iterating over all `z`, sample from distribution of `Z`; approximate the sum/integral with the sample average
- Meaning: given `x`, for each sample value of `z`, calculate `p(x, z; theta)`, divide by number of samples `k` and multiply by number of possible values `Z` can take (TODO: this is in case of sum, how to do it for the integral? Do we need to discretize?)
- Problem with this, you'll rarely get a good sample (for most `z`, `p(x|z)` will be very low)
- We would like to pick `z`s that "make sense"
- Could we use some distribution `q(z)` that produces "good" values of `z`?
- What does it mean to be good? Well, basically, we want values that are likely under `p_theta(x,z)`
- Let's see what we can derive mathematically
- Remember we are looking at likelihood `L(x, theta) = p_theta(x) = sum [p_theta(x,z)] over all z` (and that is what we want to optimize, or, more precisely, a log of that)
- _My note: Why log? Reminder: Because we want to optimize across all datapoints, which normally means product of probabilities, but if we take log of that product, we can convert it into sum of logs, which is more convenient_
- Note that we can multiply and divide `sum [p_theta(x,z)] over all z` by some distribution `q(z)` (this is always true and `q(z)` can be any distribution); which immediately takes shape of an expectation for the `Z` distributed `q(z)`, meaning we can approximate it by sample average
- For the reference, this is what we actually get: `p_theta(x) = E[p_theta(x,z)/q(z)]` under `Z~q(z)` (we got rid of sum)
- And remember, we want to maximize the log of this expression (here we just massaged our precise training objective into the form of expectation)
- By Jensen inequality, we can actually move the log inside the expectation, which will give us a lower bound for the log of expectation
- Meaning: `log(E[p_theta(x,z)/q(z)]) >= E[log(p_theta(x,z)/q(z))]` under `Z~q(z)`
- The quality of the lower bound depends on the choice of `q(z)`, if we have a good `q(z)`, maximizing the lower bound could be as good as maximizing the original expression
- It can be proven mathematically that when `q(z) = p_theta(z|x)`, the lower bound becomes the exact expression (inequality becomes equality)
- So, coming back to our choice of `q(z)`, the ideal `q(z)` is actually `p_theta(z|x)`
- Meaning, if we could sample from `p_theta(z|x)`, we could replace expectation `E[log(p_theta(x,z)/q(z))]` with sample average of `log(p_theta(x,z)/q(z))`, and optimizing that would be equivalent to optimizing the actual objective `log(E[p_theta(x,z)/q(z)])`
- All of which is just a formal way to say "let's approximate our objective by sampling `z`s that make sense"
- But how would you get to `p_theta(z|x)`? In many cases this function is actually intractable (so you can't even compute it)
- Well, we will actually not, we will use `q_fi(z)` and jointly optimize `p_theta(x,z)` and `q_fi(z)`
- We will assume `q_fi(z)` is some easy distribution with parameters `fi`, e.g. `Gaussian(mu_fi, sigma_fi)`
- We will guess `q_fi(z)` from data, so it becomes `q_fi(z|x)`
- Note that we have to use a different `q_fi(z|x)` for each datapoint (image), because the likely values for latent variables depend on what we see
- The final learning objective: `L(X; theta, fi) = E[log(p_theta(x,z)) - log(q_fi(z|x))]` under `q_fi(z|x)`
- Conceptually, the algorithm is as follows:
- 1) Initialize `theta` and `fi` for each datapoint ("somehow")
- 2) Randomly sample a datapoint `x_i`
- 2) Optimize our joint objective with respect of `fi_i`
- 4) With that `fi_i` fixed, optimize our joint objective with respect of `theta`
- Optimization is by gradient descent
- Calculating gradient with respect of `theta` is easy, gradient goes inside expectation, then we approximate the expectation by sample average
- Calculating gradient with respect of `fi` is hell, because we are sampling from `q_fi` where parameter `fi` is the thing we are calculating the gradient with respect to
- Apparently there are some tricks that work in some cases (look up "reparametrization")
- But this is still unrealistic to optimize since there is too many `fi`s, and the whole objective is non-convex, so we need more approximations
- _My note: it goes into a rabbit hole_
- **Amortization**: learn the mapping from `x_i → fi_i` using a single NN across all data points. Meaning NN spits out parameters `mu_fi`, `sigma_fi` for the Gaussian, using weights `fi`
- Amortization refers to the fact that you do most of the work during training, and have an easy way to get the `q_fi` without doing any integrals
- You would still need reparametrization trick and sampling to compute gradients
- Finally, this translates into an encoder-decoder architecture: encoder takes an image and produces parameters of a distribution of the latent variable and decoder taking that distribution and reconstructing the image

### VAE optimization summary

- `Z -> X`, `p_theta(x|z)`, `q_fi(z|x)`
- `q_fi(z|x) = Gaussian(mu_fi(x), sigma_fi(x))`
- `p_theta(x|z) = Gaussian(mu_theta(z), sigma_theta(z))`
- Encoder: NN `x_i` -> `mu_fi(x_i), sigma_fi(x_i)`, weights `fi`
- Re-parametrization: `z_i = mu_fi(x_i) + epsilon*sigma_fi(x_i)`, where `epsilon ~Gaussian(0,1)`, a single sample for `epsilon` is good enough
- Decoder: `z_i -> p_theta(x_i|z_i)`, weights `theta`
- Training objective: `max [1/N sum [log (p_theta(x_i|mu_fi(x_i) + epsilon*sigma_fi(x_i)))] over all i - D_KL(q_fi(Z|x_i)||p(Z))] wrt (theta, fi)`, where `N` is a size of a minibatch
- `p_theta(x_i|mu_fi(x_i) + epsilon*sigma_fi(x_i))` is the output of decoder, `D_KL(q_fi(Z|x_i)||p(Z))` computed in a closed form using software
- `p(z)` is a prior distribution of `Z`. The vanilla implementation of the VAE assumes `p(z) = Gaussian(0,1)`
- Also see [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
- [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae)


## Normalizing flows

- See this as well: [Introduction to Normalizing Flows](https://www.youtube.com/watch?v=u3vVyFVU_lI)
- VAEs are good, but they are pain in the ass to train, since it's very hard to compute the integral `p_theta(x) = integral [p_theta(x,z)] over dz`
- We want `p(x)` to be easy to evaluate and to sample from (while being able to describe a complex distribution)
- The key idea of normalizing flows is to map simple distributions (Gaussian, Uniform etc.) to complex ones by applying invertible transformations
- How: make `X = f_theta(Z)`, i.e. make `X` a deterministic and invertible function of `Z`
- This means for any `x` there is a unique corresponding `z`
- Note, that also means `x` and `z` have the same dimensions, so there is no compression
- And since there is no compression, `Z` is no longer a latent variable that captures some high-level meaning
- It is still a latent variable and does still capture some meaning, e.g. you can interpolate between different values of `z` to morph between different `x`, but it's hard to interpret what it does
- Essentially, this approach allows mapping images into Gaussian noise and back, hence it's normalizing
- It's called flow, since you can apply these transformations in a long chain, until you get a perfectly shaped Gaussian
- Sampling from `p(x)` means sampling `z` from `p(z)` and then computing `x` by transforming `z`, easy
- To evaluate `p(x)` you actually need to get `p(x)`. But how do you get `px(x)` from `pz(z)`?
- Since distributions have to integrate to 1, you cannot simply do `px(x) = pz(h(x))` (where `h(.)` is inverse of `f(.)`), you have to rescale the density function, using change of variables formula
- **Change of variables**: if `X = f(Z)`, and `f` is monotone with inverse `Z = h(X)`, then `px(x) = pz(h(x))*|h'(x)|`, where `h'` is a derivative of `h`
- Same expression in terms of `f'`: `px(x) = pz(z)*|1/f'(z)|`
- This is 1d case, but we want `x` to be a matrix
- The equivalent of formula in case of linear matrix transformation `X = A*Z` is `px(x) = pz(Wx)*|det(W)|`, where `W` is inverse of `A` and `det(W)` is determinant of `W`
- And `det(W) = 1/det(A)`
- That is all good, but we actually want to apply non-linear transformations, basically, we want to run `z` through NN
- Well, there is a formula for that too (it gets complicated, so I am not going to copy it here)
- And it gets even more complex if you start stacking transformations on top of each other
- But, what is important is that, even though complicated, this is perfectly calculable expression
- And, using this expression, you can derive an expression for log likelihood, which is trivial to evaluate, and which can be optimized by gradient descent
- The only caveat is, the calculation involves Jacobians, and the success of the whole undertaking depends on how fast you can calculate the Jacobians
- So what we need is to come up with some nice invertible transformations that have easily computable Jacobians (and there are some math tricks, like triangular Jacobians)
- This gives birth to multiple NN architectures that implement such transformations (NICE, Real-NVP, I-resnet, Glow, MintNet etc.)
- Auto-regression models can be viewed as flow models, if you assume `p(x_i|x_0, x_1, ..., x_i-1)` to be a Gaussian (details of this argument are omitted)
- This is the starting point for Masked Autoregressive Flow (MAF)
- The models are trained by maximizing the likelihood (to minimize the KL divergence between `Pdata` and `p(x)`)
- You could also minimize the KL divergence between `Pdata` transformed into `p(z)` and `p(z)`

### NICE

- Partition the components of variables `z` into 2 subsets `z[1:d]` and `z[d+1:n]`
- Forward mapping `Z -> X`:
- `x[1:d] = z[1:d]` (identity transformation)
- `x[d+1:n] = z[d+1:n] + m_theta(z[1:d])`, `m_theta` is NN with parameters `theta`
- Inverse mapping `X -> Z`:
- `z[1:d] = x[1:d]` (identity transformation)
- `z[d+1:n] = x[d+1:n] - m_theta(x[1:d])`
- These are functions `h(.)` and `f(.)`, don't forget that to get the distribution, you still need to multiply by the determinant of Jacobian
- Luckily, the Jacobian for this transformation is 1!
- Many of these layers are stuck together
- The final layer of NICE applies a rescaling transformation: `xi = si*zi` and `zi = xi/si`, which also has a very simple Jacobian (`diag(s)`), and determinant is `product of all si`
- So this is what we mean by saying "math tricks for applying easily computable Jacobians"
- Incredibly, if you stack enough layers like that, you can get quite impressive results
- Real-NVP is similar to NICE, but it does shifting and scaling at every level (not in the same way as NICE, but very similar)


## GANs

- So far we trained the models using maximum likelihood principle, efficiently using KL divergence for the distance `d(Pdata, Ptheta)`
- Turns out, likelihood is not always a good indicator of the quality of samples generated by the model: you can have good likelihood and bad images and vice versa
- You can construct some pathological cases where the likelihood can get as good as it gets, but the samples would be pure garbage; and vice versa
- What if we re-think the notion of a distance `d(Pdata, Ptheta)`?
- Idea: given distributions `P` and `Q`, get 2 samples from those distributions, and evaluate, somehow, how likely it is for those samples to come from the same distribution (two-sample test)
- This turns into the hypothesis testing problem, with `H0` being `P = Q` and `H1` being `P != Q`
- The usual way to do this is to come up with some test statistic `T` that compares 2 samples (e.g. comparing the mean of variance of samples)
- Reject `H0` if `T > alpha`, otherwise, fail to reject
- Since samples are random, `T` is also random, so there is room for type I and type II errors, but that is how it works
- So what we need is to come up with some good test statistic `T` and train the model to minimize `T` in the two-sample test
- Turns out, finding a good `T` in high dimensions is hard
- Guess what. Instead of hand-crafting `T`, we are going to learn it (using NN, of course). Which basically means training a classifier (aka discriminator)

### GAN architecture/objective

- The generative adversarial network have 2 components: generator and discriminator, playing 2 player minimax game
- The **generator** `G` will generate samples, starting from some latent variable `Z` (noise) and applying some transformations (similar to normalizing flows, but doesn't have to be invertible)
- The **discriminator** `D` will distinguish between "real" and "fake" samples, "real" means "coming from the dataset" and "fake" meaning "generated by the generator"
- We will use the (negative) loss of this classifier as `T`
- Why? It's not completely intuitive, but here is the explanation: the generator is basically trying to fool the discriminator to accept fake images. When training the model, we know which images are fake. So if the loss is big, i.e. if the discriminator made a huge mistake in properly classifying the image, that means the generator made a really good job faking the image
- The loss for discriminator is, naturally, the cross-entropy
- Fun fact, after all of this, if you analyze the model objective, it is actually a sum of KL divergences, aka Jensen–Shannon divergence (aka "symmetric form of the KL divergence")
- The training procedure:
- 0) randomly initialize parameters of `D` and `G`, pick a prior `pz(z)` (e.g. Gaussian noise)
- 1) sample minibatch of `m` training points `x_1`, `x_2`, ..., `x_m` from `D`, the actual dataset, these are true samples
- 2) sample minibatch of `m` noise vectors `z_1`, `z_2`, ..., `z_m` from `pz(z)`, pass them through `G` to generate `m` fake samples
- 3) update `D` by stochastic gradient _ascent_ on the cross-entropy objective (you get 1 for correctly classifying an example and take log of that, which allows summing over all points in minibatch)
- 4) update `G` by stochastic gradient _descent_ (you get 1 for every fake example that `D` misclassified and take log of that, which allows summing over all points in minibatch)
- Repeat until converges
- Turns out, using GANs, you can optimize a broader class of divergences, called f-divergences (and there exist a ton!)
- You can even optimize some more extravagant notions of distance, like Wasserstein distance
- The principle is the same, the difference is in how you formulate your optimization objectives
- GANs allow a lot of flexibility for the NN architecture, and they provide fast sampling (you just need a single forward pass)
- Unfortunately, GANs have many issues: the stabilization is unstable, difficult to estimate when it converges etc. This means you need a lot of tricks to train GANs in practice
- In particular, `D` can simply collapse to predict the same image all the time
- _My note: can you bootstrap `G` by replacing `D` by if-else looking at labeled data for the first bunch of epochs? Or you really need the `D` to be dumb and make mistakes, to allow `G` to "pick up" on those mistakes in order to learn how to better fool 'D'?_
- These problems are the reason GANs are falling out of favor


## Energy based models

- We want to keep the flexibility of GANs but still have a stable training maximizing likelihood
- We are still looking for `p(x)` that is closest to `Pdata`, and we want our model to spit out that `p(x)`
- Notice that we could use any `f(x)` as `p(x)` (and typically we would like to use NN), but it has to satisfy 2 constraints: be non-negative and sum/integrate to 1
- Non-negativity is easy to achieve: if you are using NN, you could simply square the output of NN, and voilà
- Integration to 1 is more difficult, but also the key to the model (because it enforces the zero sum game)
- Just think how difficult it would be to ensure that NN outputs sum to 1 over all possible inputs (and always, meaning regardless of how you adjust the weights during training, the constraint still holds)
- _My note: softmax only ensures all the output dimensions sum to 1, for a single input, so it's not the same_
- So we will try to do exactly that: find some family of functions `f(x)` parametrized by `theta` (i.e. trainable), ideally, as flexible as possible, and that satisfy the 2 properties of a probability distribution
- If you think about it, the requirement of integrating to 1 is, essentially, normalization: if you divide the individual output by the total volume (aka normalization constant), you would essentially achieve the requirement
- So the key is to be able to easily calculate the total volume (from `theta`), which would allow us to normalize the output of any `f_theta(x)`
- Ideally, we want to compute that value analytically, in a closed form, as a function of `theta`
- Our approach will be to carefully select `f_theta(x)` so that we know how to compute the total volume analytically
- It sounds like a difficult task, but actually, it is not
- For example, if you look at the expression for the normal distribution, the `e to the power of something` part is basically one of those functions `f_theta` and the `square root` part in the denominator is a corresponding normalizing factor
- And if you actually verify, by computing the integral of nominator over `x`, you indeed get the denominator (and that would be a formal way to do it)
- _My note: yet another way to see the normal distribution. It's fascinating how the things connect in math and how every single detail seem to have a deeper meaning_
- For the exponential distribution, there is no denominator, but, just like with normal, you can compute the integral of the expression for the exponential distribution over `x`, which gives `1/lambda`
- And turns out, this can be done to the whole exponential family of distributions (Normal, Poisson, Exponential, Bernoulli, Beta, Gamma, etc.). So, fiesta
- This leads to the definition of **energy based model**: `p_theta(x) = exp(f_theta(x)) / integral [exp(f_theta(x))] by dx`
- Basically all the same ingredients we just discussed above
- We will denote the integral part as `Z(theta)`, called **partition function**

```
p_theta(x) = exp(f_theta(x)) / Z(theta)
```

- `exp` is there by choice, allows fitting exponential family of distributions into this model, and those are very commonly present in nature (e.g. you can find them in thermodynamics etc.)
- Actually, the name "Energy based models" comes from the relation to physics, `-f_theta(x)` being the expression for energy in statistical physics, but we are going off-topic
_My note: just when I thought I understood where it was going, we are now suddenly going to change ideas, it seems. I.e. we are going to use the shape of the expression above, but seem to completely drop the requirement of being able to compute `Z(theta)` analytically and easily. I get it, we want to use NNs as `f_theta`, and then it becomes hard to come up with `Z(theta)`. I just wish the storytelling was better_
- This is an extremely flexible model, but unfortunately, sampling from it is super hard, evaluating `p_theta(x)` is also hard, which means training is hard
- It also suffers from the curse of dimensionality: computing `Z(theta)` numerically (not analytically) is `O(exp(dims(x)))`
- Is there anything that is good? :D
- Well, you can compare `f_theta(x)` and `f_theta(x')` without normalizing, and their ratio does not involve `Z(theta)` either. And many tasks only require relative comparisons (e.g. image restoration, where you need to compare the original image with the restored one)
- Supposedly, an example, image de-noising: original image `x` and the de-noised image `y` that we want to recover. We express joint probability `p(x,y)` in a form of an energy based model, where `f_theta` will be relating every pixel of `y` to the corresponding pixel of `x` + the surrounding pixels of `y`. Now we want to find `y` that maximizes `p(x,y)`. For this task, clearly, we can ignore the normalizing factor `Z` (similar to naive Bayes)
- _My note: I can see how we don't need `Z`, I don't see why would you compare 2 different `y`s. Is this not simply a gradient descent optimization, starting from some `y`? I think the point is, in many applications, the normalizing factor can be ignored, which I totally buy_
- Another good thing is that we can combine predictions of different models into a single number, by taking the product of probabilities and normalizing by the total volume across all those models
- To train, you need to maximize the log of the expression for `p_theta(x)` on the training samples (basically, maximum log likelihood w.r.t. theta)
- **Contrastive divergence algorithm** approaches it in the following way: sample `x_sample` from the model (from `p_theta`), take `x_train` from the training set, calculate gradient on `f_theta(x_train) - f_theta(x_sample)` w.r.t `theta`, take a step
- Magically, you can formally prove that this expression is a Monte-Carlo estimation for the true gradient of log likelihood, so we are all good
- But how to sample? It's hard if we don't know `Z(theta)`
- We can leverage the fact that we can compare `f_theta(x)` and `f_theta(x')` without normalizing, just start with some random `x`, try some random perturbations and keep those candidates that are more likely (actually, you occasionally still pick candidates that are less likely, seem to be some kind of exploration-exploitation tradeoff)
- This takes a very very very long time
- The improved procedure is aiming to improve the way we do random perturbations, trying to make it less random (see Unadjusted Langevin)
- This is still very expensive at training time
- This is why we need another training approaches, that don't require sampling

### Score matching optimization

- `log(p_theta(x)) = f_theta(x) - log(Z(theta))`
- To optimize w.r.t. `theta`, we really need a gradient of this w.r.t. `theta`, but what if we take a gradient w.r.t. `x`?
- This is **score function**: `s_theta(x) = grad log(p_theta(x)) wrt x`
- Score function eliminates `log(Z(theta))`, as it doesn't depend on `theta`
- The score function provides kind of an alternative description of likelihood function (you replace the actual curve height with a measure of its steepness at every point)
- **Fisher divergence** uses the score function to compare 2 distributions `p(x)` and `q(x)` (compares their scores)
- The idea is: if 2 distributions are similar, they should have the similar field of gradients
- And as with every good divergence, it is 0 when 2 distributions are the same
- **Score matching optimization**: minimize the Fisher divergence between `Pdata(x)` and `p_theta(x)`
- This, unfortunately, involves computing gradient on `log(Pdata(x))` w.r.t. `x`
- But, if you are willing to assume `Pdata(x) → 0` as `x → infinity`, you get a new loss function that only requires data
- Unfortunately, this requires computing of Hessian (same as Jacobian of gradient of f), which is still very expensive for large models

### Noise contrastive estimation

- The idea: contrast the model with a noise distribution
- This is, essentially a GAN architecture
- The generator is not trainable, generates noise (pick some distribution of noise from which you can sample efficiently)
- The discriminator is trainable and has a very specific form that is based on an energy based model, and spits out the probability (not just any binary classifier)
- To make it possible, we will treat `Z(theta)` as another learnable model parameter
- The objective and the algorithm are the same as with a regular GAN (cross-entropy loss), but now you have to optimize w.r.t. both `theta` and `Z`
- We will plug the expression for the discriminator into the cross-entropy loss to get the actual training objective

### Flow contrastive estimation

- Same as noise contrastive estimation, but generate the noise using normalizing flow model
- With this approach the generator becomes trainable


## Score based (diffusion) models

- One approach is to model `p(x)` explicitly (e.g. autoregressive model)
- Another approach is to model the sampling process without explicit `p(x)` (e.g. GANs)
- When discussing energy based models, we've seen yet another approach: use gradient of (log of) `p(x)` as a substitute for `p(x)` (aka score function)
- Many different functions can have the same gradient, so there is, in theory, an information loss. However, all these functions will be the same up to the constant, but in case of pdf, this constant can be fully recovered from the fact that pdf have to integrate to 1, so this makes it 1:1 relationship. Actually, quite cool
- Unlike `p(x)`, `s_theta(x)` does not have to satisfy any constraint, so it's potentially much easier to work with
- We already used it to train model by minimizing Fisher divergence
- This technique can be applied to auto-regressive and normalized flow models, if you just replace `p(x)` with `s_theta(x)` and train by minimizing Fisher divergence instead of maximizing the likelihood
- What is the most general model family that this can be applied to?
- We will define **score-based model** to be that general model family (will include auto-regressive and normalized flow models)
- The model: given samples `x1, x2,..., xn` from `Pdata(x)`, learn `s_theta(x)` that is a good approximation of `grad log(Pdata(x)) wrt x` (estimate that from Fisher divergence)
- For the training to be successful, `s_theta(x)` needs to be efficient to evaluate

- TODO: constraints??

- The most straightforward way to do this is to use NN (of course), but, as we seen in Score matching optimization, requires calculating Jacobian (very expensive)
- So we need some tricks to overcome this
- Turns out, by some magic of math, calculating `s_theta(x)` is much easier if you add some noise to the data (allows getting rid of Jacobians)
- **Denoising score matching** is based on matching the score of noise-perturbed distribution
- In plain English, instead of fitting a model to the data, you fit it to the data+noise. If the noise is small, this works good enough. And the worst case, you can always apply de-noising algorithm on top of your generated samples, which is relatively easy task
- Another approach: instead of solving the original problem in many dimensions, solve in one dimension (should be easy)
- **Sliced score matching**: project 2 vector fields onto random projections, and optimize for each projection (also gets rid of Jacobians)

### Generating samples

- Since all you have is `s_theta(x)`, generating samples gets a little tricky
- To generate samples, you use the fact that gradients point in the direction of local maxima
- So you can randomly initialize a bunch of particles, and let them move in the gradient field, they should settle at the local maxima
- All is cool, but in that case, the particles will actually collapse to those points. In reality, we want a "cloud" of particles that look like samples (follow the probability distribution, not all collapsing to the mean)
- So you need to follow the noisy gradients (see "Langevin dynamics sampling")
- It can be proven that, with step size going to 0 and number of steps going to infinity, the Langevin dynamics sampling produces correct samples
- Unfortunately, this doesn't work in practice :D
- One of the reason is, the regions of low data density will stay undertrained (the model will not produce good values for gradients in the areas that are far from data samples)
- And this also may take a very long time
- Another issue: Langevin distributes points between 2 local maxima equally, regardless of the height of the local maximum, when in reality you want more samples near the higher peak
- So Langevin is a bit shit


Continue with Lecture 14