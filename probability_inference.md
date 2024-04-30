# Probability, basics of inference
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [MIT 6.041 Probabilistic Systems Analysis and Applied Probability](https://www.youtube.com/playlist?list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8)

## Intro

- Reality → data → model → prediction
- Many models are possible, and not always clear which model is the best
- Separating signal from noise is classical inference problem (randomness comes from the noise)
- Given signal `S`, amplified by `a` by the medium in which it propagates, then some noise `W` is added, and you observe `X = aS + W`
- If you want to build a model of a medium, you will be interested in `a` (system identification model)
- If you know about the medium, you might be interested in reconstructing `S` by separating noise `W` from `X`
- Another classical problem is estimating unknown quantity (randomness comes from the sampling)


## Classical vs Bayesian statistics

- Classical and Bayesian statistics uses completely different philosophical approaches
- **Classical statistics**: assume that the unknown is a certain quantity that has a definite value, not a random variable, e.g. mass of an electron. Your measurements introduce error, which is random, and you are trying to remove the noise to obtain a "true" value
- **Bayesian statistics**: anything I don't know I consider to be a random variable, even numbers like mass of an electron. This statement is not about the nature, it is about my beliefs (my belief about the mass of an electron is a random value). I assume that I have some initial guess about the distribution of this random variable. After observing the data, I update the distribution of this random variable using Bayes Rule
- The rest is very similar, you measure the value through the experiment, you get some random errors, and you are trying to get the best estimation of the unknown
- In both cases, the result is an estimate, which is a random variable


## Bayesian inference

- Example: estimate the bias of a coin `theta`, given that we observe `X` heads in `n` tosses
- Classical statistician would assume there is some true `theta`, come up with an estimator, e.g. `X/n`, and argue that this is a good estimator, using, for example, the weak law of large numbers
- Bayesian statistician would assume `theta` is not exact number, but a random variable with some (prior) distribution (which allows us to condition on `theta`)
- As a Bayesian statistician, you would be required to pick a prior distribution (and justify your choice!)
- Sometimes the prior distribution is known from the system design (e.g. errors distribution that is specified by the design of a communication system)
- Sometimes you need to use your beliefs to pick a reasonable distribution, and there is no exact science to tell you how to do it
- For example, if you have no faith in the factory that makes coins, you might assume uniform distribution: any bias is equally likely
- On the other hand, if you have some faith in the factory, you might assume a normal distribution centered narrowly around 0.5, but not exactly 0.5, due to some manufacturing errors
- If you truly have no idea, there are some **uninformative priors** you could use (although this typically yields results which are not too different from conventional statistical analysis)
- By Bayes rule, `f(theta|X) = f(theta)*p(X|theta)/p(X)`, where `f`'s are PDFs and `p`'s are PMFs
- `f(theta)` is a **prior**, `f(theta|X)` is a **posterior**
- `p(X|theta)` is the **likelihood**: the probability of observing our result as a function of `theta`
- Once you pick your prior, formulas for `f(theta)` and `p(X|theta)` are known
- `p(X)` is the **evidence**, calculating it is very hard, but since it is a constant, it is usually ignored (basically, it only acts as a scale factor for `f(theta|X))`
- So people normally drop it and use `f(theta|X) ~ f(theta)*p(X|theta)`
- This can still be very complicated expression to compute, if you are not a mathematician
- But if you manage to do that, you'll get a posterior distribution, that tells you possible values of `theta` and their probabilities
- What if your boss just wants one value of `theta` (a **point estimate**), and you need to make the decision on what `theta` is?
- If you want to make the decision that is most likely correct, you would pick the value with the highest probability: **maximum aposteriory probability estimate (MAP)**
- Of course, this single answer can be misleading, as the information is lost
- Especially with continuous distributions, narrow and higher peaks are not necessarily better than lower but wider ones

### Least mean squares estimation (LMS)

- This is another way to pick a single value to report to your boss
- Imagine your posterior distribution `f(theta|X)` is uniform from `a` to `b`, and you need to pick the value `c` to report, `a <= c <= b`. What would you pick? How would you pick?
- In this situation, you might decide to come up with a value that simply minimizes a possible error
- The error is `|theta-c|`, the difference between a "true" value and the value you would pick, whatever value that is
- We will actually take an error squared, to penalize larger numbers even more
- So, mathematically, we just want to minimize `E[(theta-c)^2|X]` with respect to `c` (i.e. which `c` gives us the smallest value)
- And, if you take derivative and equal to 0, you'll find that the optimal estimate is `c = E[theta|X]`
- In case of uniform, this is just `(a+b)/2`
- You could report this number `c` together with average size of your error (which would be `Var(theta|X)`)
- If you are minimizing the error squared, you can't do better than LMS estimator
- _My note: is that the best thing to do? Probably depends on the problem_
- Nice property of this estimator: on average errors cancel each other out, i.e. `E[error|X]=0` (can be proven)
- This is good, since we want an estimator that does not have a systematic error (unbiased)
- Also, there is no systematic relation between your estimate and the error, `Cov(c, error)=0`

### My Example

- Your friend sends you a gift for your birthday, and you would like to guess how much the gift costs (`theta`)
- You know that your friend has a rule to pay somewhere between 20 and 30 EUR for a gift, no more, no less: that's your prior `f(theta)`
- You also know that the shipment is 10% of the price of an item for orders < 25 EUR, and you pay flat fee of 2 EUR on any item that is >= 25 EUR: that's `f(X|theta)`
- The delivery company slapped a sticker on a package revealing the shipment cost of 2 EUR, that's your observation `X`
- You conclude that he either paid exactly 22 EUR, with some small probability, or anything between 27 and 30 EUR, also with some probability
- So the price of your gift is either exactly 20 EUR or something between 25 and 28 EUR: that's your posterior
- That's nice, but you want to match the price of the gift, so you want to pick a point estimate
- You pick 20 it might be too low, your friend might be pissed
- You pick 28 it might be too high, your friend might be embarrassed
- You decide you want to minimize the error, so you go with LSE estimation and pick something like 26.4
- This is your best guess based on the observation

### Linear LMS estimator

- Least square error estimator is good, but may produce a complicated curve, and calculating `c = E[theta|X]` in practice may be very difficult (e.g. involve multidimensional integrals)
- Instead, we could decide to use a linear estimator, `c = a*X + b`, that models `c` as a function of `X`, and use it to produce an estimate
- Turns out, this is not a completely crazy thing to do
- And, in case of multiple data points, it actually looks more attractive
- LMS in that case would be `E[theta|X1,X2...Xn]`
- Our linear estimator is `a1*X1 + a2*X2 + ... an*Xn + b`
- We will still minimize an expectation of a quadratic error, i.e. `E[(theta-c)^2] = E[(theta - a*X - b)^2]`, with respect of `a` and `b`
- In this case, this translates to minimizing the distance from the line
- As usual, take derivative and equal to 0, this will give you the values for `a` and `b` (omitted here)
- In case of modeling signal+noise, `Xi = theta + Wi`, with all `Wi` independent and centered around it's mean, the linear estimator has a closed form solution
- And, if all distributions are normal, this expression turns into `E[theta|X]`, so it becomes the same as LMS estimator
- This give this method another interpretation: linear estimation is basically LMS pretending that all variables are normal


## Classical inference

- You assume there is a specific unknown value of `theta`, it gets mixed with some noise `N`, producing `X` with some distribution `Px`, you want to estimate `theta`
- Unlike in Bayesian statistics, `theta` is not a random variable, and the `X` is not conditional on `theta`
- **Parameter estimation**: unknown is continuous, aim at a small estimation error
- Example: determine how much effect a treatment has
- **Hypothesis testing**: unknown takes one of few possible values (discrete), aim at small probability of incorrect decision
- Example: determine whether a treatment has an effect
- Two popular methods: maximum likelihood estimation and sample mean estimation

### Maximum likelihood estimation

- The idea is to pick `theta` that makes data most likely ("fit the distribution to the sample")
- In essence, if you see 7 heads and 3 tails, you conclude that the coin is biased with `theta` 0.7
- Philosophically, you say you have several valid models for your process, and you pick one that looks the most plausible
- Mathematically, this approach is the same as using Bayesian statistics and always picking uniform as a prior
- If we have i.i.d `X1, X2, ... Xn`, measurements that come from some distribution with unknown mean `theta` and some variance `sigma` squared, we can express the probability of seeing these results as a function of `theta` and maximize that function with respect of `theta`
- Maximizing a function is the same as minimizing the log of the function, which is what usually done in practice, as it turns out to be easier
- Minimizing a function is done by taking a derivative (and equaling to zero)
- We need to evaluate how good this estimator is
- Ideally, we would like to have an estimator that is close to a true value and has no systematic error (errors cancel each other)
- In general, maximum likelihood estimators are biased, but the bias tends to disappear with large amount of data (can be proven mathematically)
- It also can be proven that maximum likelihood estimators provide an estimate that converges to a true value with large amount of data
- If we look at the expectation of a **mean squared error (MSE)** of a such estimator `(error = (estimation of theta - theta))`, we can see that it consists of two part:
- `E(error^2) = Var(estimation of theta) + (E(error))^2 = variance + bias^2`
- So it has two contributions: **variance** is a measure of spread of your estimation, **bias** is the systematic error of the estimator
- Ideally, we want to keep both variance and bias as small as possible

### Sample mean estimation

- Setup: you have a i.i.d `X1, X2, ... Xn`, measurements that come from some distribution with unknown mean `theta` and some variance `sigma` squared
- You assume the model `Xi = theta + Wi`: every measurement is a combination of a true value + noise
- You come up with an estimator `(X1, X2, ... Xn)/n`, which by the weak law of large numbers converges to a true mean
- When you go to your boss, you don't just report the value (the point estimate), you also want to give him some more information: a **confidence interval**
- We want to report an interval `theta +- theta'` such as the real `theta` is in that interval with a given probability `alpha` (confidence)
- Philosophically, the statement is not about probability of `theta` being in that interval, it is about the probability of a given interval to capture the true `theta` (because `theta` is not random)
- Typical values for alpha: 0.05 (the most common), 0.1, 0.01
- Here we use the central limit theorem
- We "massage" the expression `(theta - theta')` to look like `Zn`, replace with Normal distribution `Z`, and use `Z` to construct lower and upper bounds for the interval of a desired confidence
- Using alpha=0.05 we get `+- 1.96*sigma/sqrt(n)`
- More generally, you need `+- z*sigma/sqrt(n)`
- The slight problem is: you don't always know `sigma`
- One approach is to take the most conservative value (i.e. find an upper bound), and report a confidence interval that is bigger than necessary
- Another, easier, approach is to estimate `sigma` from our data, using an estimate for the mean. When `n` is large, this all works

### Linear regression

- This is a model for parameter estimation, built upon maximum likelihood estimation, when the dependency between an unknown parameter and the data is linear
- This kind of modeling is a subject of a separate course
- It is a good idea to report the estimation together with confidence interval
- You can also report and calculate `R^2`: the measure of explanatory power (how much data can be explained by this model comparing to how much can be attributed to random factors), normally the software packages report it
- Linear regression cannot make statements about causal relation between parameter and estimation

### Hypothesis testing

- You are considering two potential distributions: **null hypothesis** (i.e. a default hypothesis) and some alternative hypothesis
- You want to know which one it is: do we accept or reject the null hypothesis
- In general, the 2 distributions overlap, think 2 gaussians next to each other
- You divide the space of all possible results of an experiment in 2 regions: rejection region and acceptance region
- One simple way to decide between 2 regions is to use some kind of threshold
- Using this model, I can make 2 types of errors: `FP` and `FN`
- You would like to minimize both kinds of errors, unfortunately there is a tradeoff (ROC curve)
- So we usually fix the rate of one category of errors, and allow the other to move (by projecting on ROC curve)
- We find an optimal value of a threshold so that it maximizes the likelihood
- We do that by finding the likelihood of both hypothesis and calculating the ratio
- You can prove mathematically that maximizing the likelihood produces the optimal ROC curve
- See Statistics for more detailed explanation