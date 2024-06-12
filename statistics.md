# Statistics, theoretical foundations
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [MIT 18.650 Statistics for Applications](https://www.youtube.com/playlist?list=PLUl4u3cNGP60uVBMaoNERc6knT_MgPKS0)

## My takeaways

- The essence of statistics is replacing expectations by averages


## Statistics vs Probability

- Probability knows what to do with randomness when facts (probabilities) are given
- Example: knowing that drug is 80% effective, we can predict that on average 80 out of 100 patients will be cured, and at least 65 will be cured with 99.99% chances
- Statistics goes backwards, extracting those facts (probabilities) from data
- Example: knowing that 78/100 patients were cured, conclude that we are 95% confident that drug will be effective on between 69.88% and 86.11% patients
- Not everything is truly random. Sometimes we call things "random" just because the phenomenon is too complex to understand
- So randomness is a way of modeling lack of information, with perfect information about the universe we would be able to accurately predict the outcome every time
- Complicated process = simple process + random noise, and the random noise is everything you cannot understand
- Good modeling is choosing plausible simple process and noise distribution, and it requires a lot of domain knowledge


## Typical problem we are trying to solve

- Is there a preference for couples to turn their heads to the right when kissing?
- Let `p` be a proportion of couples of the whole population that turn their heads to the right when kissing
- This is a true number, but it is unknown
- We are going to run an experiment to estimate `p`
- `n` is a size of a sample
- `p'` is an estimation of `p` using `b` samples
- Imagine we collect pictures of 120 couples kissing at the airport
- Are couples kissing at the airport a good representation for the whole population?
- Let's say 80 out of 120 turned their heads to the right, so `p' = 64.5%`
- Is that enough to conclude the preference? Would 80% be enough?
- Is number `n` big enough? How large should the sample be?
- What numbers would we expect to see should we try another study again?


### Modeling

- To evaluate the study and make sense of the numbers, we need to understand what random process the data is coming from
- For `i = 1, ... n`, `Ri = 1` if `i`-th couple turns to the right, and `Ri = 0` otherwise
- Our estimator of `p` is `p' = 1/n * sum(Ri) for all i` (basically, `p'` is the average)
- Estimator is a machinery, or model (or formula), and it produces estimate (a number)
- What is the accuracy of this estimator? We want our estimator to produce a stable number across different experiments (low variance), but also produce an estimate that is close enough to the actual number (low bias)
- As a part of our modeling, we make assumptions about our observations `Ri`:
- Each `Ri` is a random variable (due to lack of information)
- Each of `Ri` is a Bernoulli with parameter `p` (assuming our population is homogeneous; i.e. there is actually some true `p` that is the same across the whole population)
- All `Ri` are mutually independent (one couple does not influence another)


### Inference

- Take Bernoulli `Ber(p)`
- We know that for Bernoulli, `E[Xt] = p`, `Var(Xt) = p(1-p)`
- We try to estimate `p` by the sample mean `Mn = (X1 + X2 + ... + Xn)/n`
- In the central limit theorem section of probability, we defined `Sn = X1 + X2 + ... + Xn` and `Zn = (Sn - E[Sn])/sigma(Sn)`
- For Bernoulli, `E[Sn] = np`, `Var(Sn) = np(1-p)`
- So, in case of our `Ber(p)`, `Zn` can be re-written as `Zn = (Sn - np)/sqrt(np(1-p))`
- The central limit theorem allows replacing `Zn` with `Z` for a large n (>=30)
- This gives a shortcut for getting `P(|Zn|>epsilon)`
- This, massaged, allows determining `n` for a desired confidence of having maximum error `epsilon`
- Hoeffding's inequality is another shortcut, that works for any `n`
- It provides an upper bound on `P(|Sn - E[Sn]|>=epsilon)`
- Same as the central limit theorem, it removes dependency on `p`
- The upper bound might be several factors away from the actual value, and this is the price to pay


## Parametric inference

- Let's say we are interested in finding out the number of siblings in the population
- Let's say we interviewed 20 people who are students at MIT (statistical experiment), obtaining a sample of `n` values `X1, ..., Xn`
- To be able to apply statistical methods, those should be independent and identically distributed, which is achieved if we draw randomly from the same population (and don't mix number of siblings with number of oranges)
- The end goal of statistics is always to find the distribution that accurately describes the population parameter
- We can't know the actual distribution, so we will model
- Model should be simple enough to model, but complex enough to be useful
- We will make some assumptions, e.g. max number of siblings is 20
- One way to model the distribution is to use sample data to build PDF
- But in that case, we will need a table with a value for each possible number of siblings, which is 20 rows
- Also, since our sample size is very small, it's very unlikely we will have enough representation for each number of siblings, so there will be many zeroes
- Another way is to assume the distribution is Poisson (or any other known distribution) with parameter `theta`
- This will require only 1 parameter to learn, and might actually be better representing missing values
- The parameter should be derived from what we are actually trying to find out
- This model can be wrong, but it's not that important. What is important is whether the model is useful
- We assume there exists some true parameter `theta`, and we want to find it
- The model should produce the estimated value of `theta` that is close enough to the real `theta`
- The estimator is consistent if it gets closer and closer to `theta` as you collect more and more data
- We will evaluate the estimator by the quadratic risk (expectation of error squared) = `bias^2 + variance`
- Ideally, the estimator should be unbiased and has low variance
- There are multiple different ways to come up with an estimator: take sample average, use some constant, use the first value in the sample etc.
- In order to pick "the best one", you need to be able to understand the performance of these estimators in terms of bias and variance, and essentially, quadratic risk
- We can plot the quadratic risk for each of estimators
- The risk is actually a function of `theta`
- Instead of reporting just one number, we could report a confidence interval
- The boundaries of the confidence interval come from random observations, so they are also random
- You choose the width of the confidence interval so that the probability of the interval to contain true value of `theta` is `1-alpha`
- You can derive the confidence interval doing some math
- The confidence interval should not depend on unknown parameter
- If such dependency arises, we can use the upper bound for the unknown parameter or replace that with an estimation
- The meaning of confidence interval is quite tricky to grasp. The confidence is not in the parameter, but in the method. Given 5% confidence level, if you repeated the calculation many times, about 95% of times your confidence interval would actually capture the true value of a parameter. However, in about 5% of cases, you would miss it
- So it actually works best if you are conducting thousands of experiments, and you want to put a bound on a number of times you might be wrong
- And if you just built a single interval, all you know is that your parameter is either in or it is not


## Maximum likelihood estimation (MLE)

- MLE is proposed by Fisher, the father of the modern statistical science
- _My note: this is the most upside down explanation, but in some weird way, the most logical, as it follows, at every step, from very simple definitions by applying some mathematical transformations_
- Imagine we've drawn a sample with some population with some distribution with true parameter `theta*`
- As a statistician, your goal is to come up with some candidate distribution with `theta^` that is "not too far" from the true one
- We formalize "not too far" as follows: whatever value `x` you pick, `p^(x)` should not differ from `p*(x)` more than some value called **total variation distance**
- Following from this statement, the total variation distance is the max difference between `p^(x)` and `p*(x)`, for any `x`
- By some magic of math, total variation distance can be calculated as a half of all distances from `p^(x)` to `p*(x)`, over all `x`
- This has to do with constraints on `p(x)`, they both must total to 1, so their difference always totals to 0, add some more steps and that leads to the formula. Whatever
- And, of course, all the same reasoning applies when we are talking about PDFs
- Total variation distance is actually a distance from the formal point of view, as it satisfies certain properties of a distance (e.g. always positive, `=0` when `theta^ = theta*` etc.)
- _My note: yes, math has a strict formal definition of a distance_
- So you need to build an estimator that allows you to come up with `theta^`
- However, it is super difficult to build such an estimator, since we don't know `theta*`: how would we minimize distance from unknown value?
- So total variation distance, even though it is a true distance, is not super helpful
- This is why, instead of using total variation distance, we will use a "surrogate": the "Kullback-Leibler divergence" (KL divergence)
- Unlike total variation distance, KL divergence is not actually a distance, formally speaking (and this is why it is called "divergence")
- KL divergence is a `sum of p^(x)log(p^(x)/p*(x))) for all x`
- _My note: as usually, KL divergence is not pulled out of the thin air, at least not completely, there is some relation with entropy and so on, but those deeper explanations are made in non-human readable language, so I don't get them. Regardless, KL divergence is partially pulled out of the thin air, because out of many possible forms this function could have, we are choosing the one with log in it, for a very specific reason that may become clear later..._
- What is good about KL divergence (and what kind of justifies it) is that it is `=0` when `theta^ = theta*`, just like the distance, so minimizing KL divergence is going to have the same effect as minimizing distance
- What is even better, is that KL divergence has a form of an expectation of a function `log(...)` (since it has that function multiplied by `p(x)`)
- This is, apparently, a fiesta for a statistician, since you now can replace expectation by an average estimated from the sample (by law of large numbers), and minimize that
- But by average of what?
- To answer that, we'll first do some transformation, and rely on some magic properties of the log function (the logarithm of a fraction is equal to the logarithm of the numerator minus the logarithm of the denominator)
- Turns out, we can minimize this expectation by maximizing an average of `log(p^(x)) over all x` (i.e. logarithm of the denominator; since logarithm of the numerator does not depend on `theta^` and acts as a constant shifting factor)
- Also, instead of maximizing average, we can maximize `sum`
- Sum of logs is a log of a product
- Log is a function that is always increasing, so maximizing log of a function is the same as maximizing a function
- So our `theta^` is `argmax of [product of p^(x) over all x]` (under distribution with parameter `theta^`)
- Interpretation: we are trying to find `theta^` such as to make the observed values the most likely
- And `[product of p^(x) over all x]` is "likelihood" (and it is conditioned on `theta^`)
- To elaborate on interpretation even more, from probability, we remember that the joint PMF/PDF is simply a product of individual probabilities
- So `[product of p^(x) over all x]` is really a joint PMF/PDF for all `x`
- Adjusted interpretation: we are trying to find `theta^` such as to make each of the observed values the most likely
- Maximizing an arbitrary function can actually be tricky: you take a derivative, equal to zero and this gives you all the candidates, and you need to check all of them, and there can be millions of candidates
- But luckily, functions we use in stats are are strictly concave, so there is just one candidate (unique maximum)
- You still need to find it, and even if it's easy to write an equation (`f'(x) = 0`), solving it can be super hard
- This is why it is usually not solved in a closed form, but algorithmically (and that is what ML optimization methods are all about)
- And with strictly concave functions, that is somewhat easy: whatever value `x` you pick, `f'(x)` will indicate the direction you need to move to get closer to `f'(x) = 0`. You still need to decide how much you need to move
- Also, when you have multiple model parameters, the function becomes multidimensional
- And for all the previous discussion, we assumed independence of observations, however, likelihood can be defined even in case of dependent observations
- So KL divergence as a function of `theta^` has a concave shape and has minimum of zero at `theta^=theta*`. Since we are approximating the KL divergence by a function that depends on a sample. Problems begin with KL divergence functions that have a very flat curve at the bottom, since even the close approximations may produce larger estimation error. So depending on distribution, the quality of our estimator for `theta^` can vary, and this can be expressed through the **Fisher information**
- MLE is unbiased, and the variance depends on the Fisher information
- In a way, Fisher information (actually, its inverse) plays the role of variance for the MLE
- The transition from average to expectation happens when we pick a function to approximate KL divergence, the likelihood already deals with the sample data. That means when you go from probability to likelihood, you fix your observations as `X`. There is no substitution of expectation by average required

### Maximum likelihood estimation for Bernoulli

- _My note: to reiterate, one not completely obvious thing that we did, is that we started from the notion of a distance, and we found a something that behaves like a distance (for our purposes) and expressed through something that has a form of a likelihood, and thus, have seen, that minimizing distance has the same effect as maximizing likelihood. Which all makes sense intuitively. And for the conversation below, it seems that we kind of skip the part of the discussion about the distance and simply start from likelihood, and maximize that. But we need to remember, that there is always a relation to distance in the background_
- For Bernoulli, as we know, `P(X=x) = p^x*(1-p)^(1-x)`
- By definition, likelihood `L(x1, ..., xn,p) = P(X1=x1, ..., Xn=xn|p)`
- Now we start transforming this, `P(X1=x1, ..., Xn=xn|p) = product P(Xi=xi|p) over all x` (assuming independence) = `product of (p^xi*(1-p)^(1-xi)) over all x`
- And this is a likelihood for Bernoulli
- You have to maximize it with respect of `p`, i.e. finding a `p` that maximizes this expression (`x1...xn` are not moving)
- `p^ = argmax L(x1, ..., xn,p)`
- In practice, we maximize the log likelihood, it is normally easier (which is kind of taking the step-up towards KL divergence)
- We take the first derivative of the log likelihood, equal it to 0 and solve for p
- And if we do that, we'll find that `p^` is `X bar` (the sample mean)
- _My note: this is kind of a trivial conclusion, but the path how we got here is much more important. The sample mean is not simply some "common sense" way to estimate p, it can be derived from the first principles and turns out to be a MLE_
- _My note: this would be a point estimator, and you might want to actually construct the confidence interval, and looking at how it is done, it is all consistent_


## Method of moments (MoM)

- Weierstrass approximation theorem, in essence, says that "Continuous functions can be arbitrarily well approximated by polynomials"
- Now, again, we will be looking at estimator `theta^` for true parameter `theta*`
- If we find such `theta^` so that `[integral of h(x)p*(x) by dx] = [integral of h(x)p^(x) by dx]`, then we found a good `theta^`
- This is just kind of one of the million statements we could make about `theta^` and `theta*`
- What is special about it, is that `[integral of h(x)p*(x) by dx]` has a form of expectation (of `h(x)`)
- So (and what statisticians do) it can be replaced by the average
- This way `[integral of h(x)p*(x) by dx]` converts to `[1/n sum of h(Xi) over all i]`
- So now we just need to pick such `theta^` that the equality would hold for all possible functions `h`
- There is an infinity of such functions, so unfortunately, not doable
- Instead of trying all `h`, we could use the theorem and use polynomials that approximate `h`
- It can be shown that, instead of looking at all functions `h`, and even all polynomials that would approximate those, it would be enough to look only at d polynomials of form `X^k`, where `k = 1,...,d`
- So we are searching for such `theta^` so that `[1/n sum Xi^k for all i] = [integral of x^k*p^(x) by dx]`
- The expectation of `X^k` under `theta` is the `k`th moment of `p(x)` under `theta`
- What is `d`? The theorem does not tell us what `d` should be
- But what it does tell you is this: if you go to `d` "large enough", you'll be able to approximate your `h` "up to epsilon"
- _My note: and this is probably where all the computational methods kick in_
- Turns out, as a rule of thumb, you need as many moments as many parameters you need to describe the distribution
- So, in practice, `d` is often 1 or 2
- So, we decide on `d`, compute moments and recover `theta` from the equation
- Mom is one of the oldest methods for deriving point estimators. In general, the MLE is more accurate than MoM
- Sometimes people initialize MLE using MoM, this can be helpful when likelihood has many local maxima


## Parametric Hypothesis Testing

- If we look at the sample mean, we know, by CLT, that it is going to be within `sqrt(n)` from the true mean
- So the sample mean is going to fluctuate, just by randomness
- The bigger the sample, the less fluctuation
- We want to be able, however, to use `X bar` to make conclusions about true mean, whether it is `=`, `<` or `>` than hypothesized mean
- Clinical trials: test group + control group
- You may be drawing from the Poisson, but your sample means are normally distributed by CLT
- _My note: skipping the basics, since already covered by notes on Statistical Inference_
- What is important is that you are using a sample statistics to make a decision
- You calculate the statistics using an estimator, e.g. MLE
- You understand how that statistics should be distributed under `H0` (and always under `H0!`), given that you are using a certain estimator
- The estimator affects the distribution through it's bias and variance
- This distribution should not depend on any unknown parameter (asymptotically, i.e. in the limit). Usually this is achieved by applying CLT
- Then you see how likely it would be, under `H0`, to see the value you calculated from the sample
- And this is how you build any test (e.g. Z test) from first principles


## Chi-squared distribution

- This about throwing the dart at the target
- You can deviate left and right, but also up and down
- Assume those 2 deviations are independent (cause by 2 separate phenomena)
- Assume those 2 deviations are both distributed as Gaussian (2 different Gaussians)
- Now, your deviation from the center is, by Pythagoras, `sqrt(dx^2 + dy^2)`, and it is a random variable
- The deviation from the center will be distributed as Chi-squared with 2 degrees of freedom
- Chi-squared is also what you obtain if you use MLE with `d` parameters as an estimator for the (multidimensional) statistics, and you derive your test from that (Wald test)
- Wald test can also be used to test jointly multiple hypotheses on one-dimensional statistics
- There is no "standard Chi-squared" distribution, you have a completely different curve for each value of `d` (degrees of freedom) and sample size


## T distribution

- T distribution can be derived from Gaussian and Chi-squared
- T distribution is `Z / sqrt(V/d)` distribution, where `V` is Chi-squared with `d` degrees of freedom (two distributions have to be independent)
- And turns out, this is what happens when, instead of using true variance, you estimate variance from the sample (magic of math)
- So the T distribution is not some kind of random adjustment based on heuristics, it actually follows from math
- The difference only matters at small sample sizes (<50)
- So T distribution is actually used for hypothesis testing in practice, as in most cases you don't know sigma
- T distribution comes from Guinness factory


## Likelihood ratio test

- If your `H0` is that `theta = theta0` and `H1` that `theta = theta1`, you can just calculate the likelihood for both `theta`s and see which one fits the data the best
- This is the basis for the method
- Your test could be as simple as `L(X1,...Xn,theta1)/L(X1,...Xn,theta0)>1`
- The only thing is, we want to control for `alpha`, and there is no knob that allows us to do that
- This is why we do `L(X1,...Xn,theta1)/L(X1,...Xn,theta0)>C`, and find the `C` that would allow us to have a desired `alpha`
- Mathematically, this test is the best you could come up with
- However, in practice, you'll never find yourself in a situation when you have 2 candidate numbers
- So this method is just used to derive some more practically applicable one
- _My note: and then it all went into some abstract math that I eventually stopped following_


## Goodness of fit

- Goodness of fit is testing if the data comes from a particular distribution
- This is what you need to do, for example, to check that your data comes from Gaussian when applying t-test
- This is an example of non-parametric test
- What you need to do is to compute CDF and look at it
- You can build an **empirical CDF** from data, "the data version of CDF"
- You express CDF through expectation and use the usual trick of replacing it with an average (by law of large numbers)
- Essentially, for each `t`, you calculate the proportion of the data that is below that `t`
- There are tons of math that can be applied to prove that empirical CDF is a good estimator for CDF
- Usually all the goodness of fit tests are formulated in terms of CDFs, because we have an empirical CDF


## QQ plots

- QQ plots allow to visually compare the distribution of your sample with some known distribution
- This is a quick and easy test that you may want to do all the time
- You normally use software for that
- For normal, it should look like a straight line with 45 degree slope
- Right tail above the 45 degree line: right tail of the sample distribution is heavier than the Gaussian
- Left tail above the 45 degree line: left tail of the sample distribution is lighter than the Gaussian


## Regression

- Try to predict one variable based on another variable
- Typically functions like `y = a + b*x`
- We believe that the relation is truly linear and of the form `a + b*x + some noise epsilon`
- "Noise" is everything we don't understand about this relationship
- We have data, and we are going to search for the most likely line that explains the data
- Multivariate regression: `a + b1*x1 + b2*x2 + ... + bn*xn`
- From the statistical modeling point of view, we need to decide what distributional assumptions we are going to put on the epsilon, estimate `a` and `b`, and then make some inference (e.g. confidence regions for `a` and `b`)
- Example in economics: demand and price
- How do we estimate the goodness of a candidate line?
- We need a function that is big when the fit is bad, and small when the fit is good
- There are many ways to measure the distance, but one that is convenient is the vertical distance `[y - (a + b*x)], squared`
- As a statistician, I am interested in the expectation of this value, `E([Y - (a + b*X)]^2)`
- So our error function will be one that maps `(a, b)` into `E([Y - (a + b*X)])^2`
- Now we want to minimize it in respect to `a` and `b`
- After usual stuff, i.e. taking the derivatives and equaling to 0, we arrive at `b = Cov(XY)/Var(X)`, which makes sense (the math trip lands)
- Also, what comes out of it is `E(Y) = a + b*E(X)`, which also gives some insight into the method
- Now we'll consider noise: `Y = a + b*X + epsilon`, express the epsilon as `[Y - (a + b*X)]`, and find the `E(epsilon)`
- We want to know what our belief [that minimizing the square error gives us correct values for `a` and `b`] implies for the noise
- We'll see that the necessary conditions for the noise are: `E(epsilon) = 0`, `Cov(X, epsilon) = 0`
- One way to assume it is to assume that `X` and `epsilon` are independent
- At this point, you don't have to assume `epsilon` is Gaussian
- All of this is purely theoretical, but in practice, we don't know what `E(X)`, `E(Y)`, `E[XY]`, `Cov(XY)` or `Var(X)` are
- So we use the usual trick and replace expectations with averages
- Usually `x = [x1, x2, ... xn]`, so you have to do the same thing with vectors
- And normally we will use `Xi = [1, x1, x2, ... xn]` so that I can get to `Yi = Xi*W + epsilon_i`
- You do all the same things as before, but with vectors and matrices
- Using values estimated form data allows you to recover `W`, which allows you to "de-noise" the data
- Turns out, to solve the resulting linear system of equations, you need the number of datapoints to exceed the number of features
- In reality, in many modern research situations, the number of features exceeds the number of datapoints by far. This then becomes a subject of High-Dimensional Statistics, but that goes beyond this course
- The least square estimator we used turns out to be MLE estimator for `W`, if you assume epsilon distributed as Gaussian
- Once we figured out the method for regression, we could do confidence intervals and hypothesis testing
- Typical test is `H0`: `Wj = 0` and `H1: Wj != 0`
- Why does this matter? If `Wj = 0`, this means the feature `j` does not affect the value you are trying to predict
- You can imagine trying to understand whether the gene `j` affects the certain phenotype property
- Under `H0`, `Wj^` is distributed with mean 0 and variance due to noise, which gives you the recipe for hypothesis test
- The statistical packages usually can do this for you, showing the p-value next to each of your features
- Note that if you have thousands of features, 5% error on each of them will eventually accumulate
- We want to control for type I error for all features, so we need to pick the significance level of `alpha/p` where `p` is the number of features (Bonferroni correction)
- You can think about more sophisticated tests, e.g. `H0`: `W2 + W3 = 0`, and you could construct that as well. This is not usually provided by statistical packages, but the machinery is the same as in case of `Wj = 0`


## Bayesian Statistics

- Contrasting with frequentist approach, which is:
- a) Observe data
- b) Data is generated randomly (by nature, by design of the survey etc.)
- c) Make assumptions about the process that generated this data (Gaussian, Poisson etc.)
- d) There is some true parameter that we don't know, but we are interested in
- e) We want to find it: either estimate it or make a hypothesis about it
- Typically estimated using MLE
- Bayesian approach:
- a), b), c) all the same
- d) We have a prior belief about the parameter
- e) Using the data, we want to update that belief and transform it into a posterior belief
- The prior belief may play smaller and smaller role as we collect more data, but if we don't have enough data, we can make use of this belief
- So the bayesian approach is especially useful when you don't have enough data, but you have some prior belief
- For example, `p` is a proportion of women in the population
- We collect a sample, and that is `Ber(p)`
- A frequentist would simply estimate `p` from sample and do some hypothesis test `H0(p = 0.5)`
- As a bayesian, I may have some initial belief that `p=0.5`, and I want to include that belief in my statistical procedure
- Depending on the strength of my belief, I want to give less or more weight to this belief
- In many ways (though not always), this belief acts just like one extra observation
- As a result, I want to update my belief based on evidence
- As bayesian, I treat any unknown value as random variable
- We want continuous distributions as priors, so instead of Bernoulli people use Beta distribution
- If you have the same family of distributions as a prior and as a posterior, the prior and posterior are then called **conjugate distributions**
- Beta gives Beta and Gaussian gives Gaussian
- `Posterior = Likelihood * Prior / (probability of data, constant in respect to theta)`
- So what I do is updating my likelihood (which is MLE) based on my prior, weighting it by prior
- (probability of data) is an integral of `p(x1, x2, ... xn|theta) over all theta`, not easy to understand what it means, hard to compute, however, it acts as a normalizing factor so that the posterior integrates to 1
- The computation is simply multiplying 2 PDFs
- So, how do you pick your prior?
- Or, what if you have absolutely no idea?
- You can pick a prior that has no information, one of so-called "uninformative priors" (uniform)

### Bayesian confidence regions

- The bayesian approach produces an entire distribution, but we might be interested in a region of that distribution where most of its weight (`1-alpha`) is concentrated (can be several disjoint regions)
- So the confidence region is a statement where the areas of high probability are in your distribution
- The way the bayesian confidence region is constructed is by moving an `y`-intercept from `+infinity` down, capturing more and more of the probability
- Bayesian confidence regions are different from frequentist confidence regions (intervals)
- Think about time when you don't have a luxury of repeating an experiment thousands of times
- Canonical example is finding a sunk ship: you have just one single ship to find, and you want to put your data to the best use
- Bayesian statistics allows you to put a lot of prior knowledge into the model: the weather, the currents, the wind etc.
