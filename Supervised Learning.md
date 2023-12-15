# Supervised Learning

## Correctness

- **Accuracy** is the fraction of correct predictions = `correct / total`
- Having good accuracy is not enough! In a case of a very rare event, you might be able to predict with super high accuracy that it will not happen, but it's not useful
- For example, if you are predicting whether the person has cancer, the model that simply says "You don't have cancer" to everyone will have more than 99.5% accuracy. However it will miss all the people who do have cancer - so it's useless
- You need to look at **precision** and **recall**, the values that are computed from the confusion matrix
- **Confusion matrix** contains numbers for true positives, false positives, true negatives and false negatives, in the following form:
```
  [[tp, fp]
   [fn, tn]]
```
- **Precision** measures how accurate our positive predictions were = `tp / (tp + fp)`
- **Recall** measures what fraction of the positives our model identified = `tp / (tp + fn)`
- Usually the choice of a model involves a trade-off between precision and recall
- A model that predicts "yes" when it's even a little bit confident will have a lot of false positives, but very few false negatives. So it will have a high recall but a low precision
- A model that predicts "yes" only when it's extremely confident will have very few false positives, but a lot of false negatives. So it will have a low recall and a high precision
- In other words, saying "yes" too often will give you lots of false positives; saying "no" too often will give you lots of false negatives
- Depending on the situation, you might prefer false positives or false negatives
- In case of the patient diagnosis, false positives are better, since we don't want to send a sick person home, but we are OK with doing more tests on a healthy person that is mistakenly put in a positive group. So we want a high recall: "let's make sure we catch it, whatever the cost"
- In case of the spam mail classification, false negatives are better, since we don't want to accidentally miss an important email from someone we know just because it went to the spam folder, but we are OK to see a spam email in the inbox from time to time. So we want a high precision: "if it is classified as spam, it'd better be spam"
- We can combine values of recall and precision in one value. One way to do it is to calculate **F1 score**: a harmonic mean between recall and precision = `2*r*p / (r + p)`
- F1 score reaches its best value at 1
- F1 score treats recall and precision as equally important
- If we want the score to favorite recall or precision, we can calculate **F beta score**, where beta is a coefficient that drives the score up depending on recall or precision
- Lower beta favors precision, high beta favors recall. Coming up with a good value of beta is a matter of experience and expertise
- **ROC curve** is a chart of (`tp / all positives`) to (`fp / all negatives`). The closer the area under the curve to 1, the better the split is (see appendix for details)
- Sometimes, instead of precision and recall, people use sensitivity and specificity
- **Sensitivity** (true positive rate) is the same as recall: `tp / (tp + fn)`
- **Specificity** (true negative rate) measures what fraction of the negatives our model identified: `tn / (tn + fp)`. It's kind of like "recall on negatives"
- A highly sensitive test rarely overlooks an actual positive. Some of the positives may be false positives, but you surely catch them all
- A highly specific test rarely registers a false positive. It might not catch all positives, but once it reports one, you can trust it is one


## Overfitting and underfitting

- **Fitting** is adapting the model parameters to produce results with the higher degree of "correctness" (i.e. with desired score for precision / recall)
- **Underfitting** is producing a model that doesn’t perform well even on the training data
- **Overfitting** is producing a model that performs well on the data you train it on but that generalizes poorly to any new data
- Overfitting is caused by 2 factors. One is learning noise in the data. Another is learning to identify specific inputs rather than whatever factors are actually predictive for the desired output (memorizing the input)
- So the simpler model may do some mistakes on the test data, but we can think of them not as mistakes but just result of noise in the data (see probabilistic interpretation below). Looking at it this way, it is actually good when the model makes "mistakes" in case of noise
- The most fundamental approach to detect and avoid overfitting involves using different data to train the model and to test the model (typically 70/30)
- Your test data should have the same distribution of the feature values as the train data. You should verify it
- There are 2 possible sources of errors in the model predictions: bias and variance (closely related to bias and variance of statistical estimators, but used in less formal way)
- **Bias** is an error from erroneous assumptions in the learning algorithm. It works as a measure of the quality of prediction on a training set
- An example of an algorithm that usually has high bias is linear regression
- **Variance** is an error from sensitivity to small fluctuations in the training set. It works as a measure of how different the quality of predictions is on two training sets randomly chosen from the same data
- An example of an algorithm that tends to have a high variance and low bias is a decision tree (especially decision trees with no early stopping parameters - they can adapt to any training data)
- Underfit model will have a high bias and low variance. Overfit model will have a low bias, but high variance
- Ideally we want low bias and low variance :) Unfortunately, it is typically impossible to achieve both simultaneously. So you need to find a tradeoff between the two
- If your model has high bias then you can try adding more features
- If your model has high variance then you can try removing features
- Another way to lower the variance is to add more data. Holding model complexity constant, the more data you have, the harder it is to overfit
- However more data won't help with bias
- More complex models need more data otherwise they overfit. Simple models might be just enough for your case
- One way to prevent the high the model complexity is **regularization**, see example in linear regression
- Many models allow to adjust the hyperparameters, for example, learning rate in case of linear regression or maximum tree depth in case of decision tree. These hyperparameters usually help to fine-tune the model in the way to combat the overfitting
- You might decide to choose an optimal model from a variery of existing algorithms and optimal values of hyperparameters, by repeating the training with different models and values and comparing the results. However, in that case you need to use validation set, and not the test set!
- If you are using your test set for tuning, you are overfitting again. Your "optimal" model is going to be tuned to the test data, but not necessarily to the real world data
- Basically, you are not allowed to make any decisions about the model based on the test set (but it is OK to report the performance on the test set over time)
- One way to create a validation set is to split your input data into 3 subsets instead of 2 (training + validation + testing)
- Typically 60/20/20, but on very large datasets (million of datapoints) you typically would not need validation/test sets that big, so you could go to 90/5/5; unless you are looking for the tiny differences between algorithms (0.01%), in which case you need larger validation sets
- Using a separate subset for validation is called **hold-out cross validation**
- The disadvantage of hold-out cross validation is that you are not making the best use of your data: the datapoints in the validation set are "lost" for training and vice versa. This is especially critical for small datasets
- Alternatively, you can use **k-fold cross validation**, which means that every time you train the model, you choose 1/k of your training data to be a validation set
- You split your training set into k folds, find the optimal value for a hyperparameter for every fold, and use it to calculate the value that is good for all folds
- k = 10 is a popular choice, k = 5 or k = 20 are also used
- The disadvantage of k-fold cross validation is that it is computationally expensive (you are evaluating each model k times), so it is mostly used on smaller datasets
- The extreme version of k-fold cross validation is leave-one-out cross validation, which can be used in case of very small datasets (<=100)
- You could use similar approach to find best subset of features, as a way to reduce a number of features for your model. This is called **feature selection**
- One way is to start with a small subset of features and try adding more, one by one. Alternatively, you can start with all features, and try removing one by one. Both methods are computationally expensive
- There are some cheaper methods to select a feature subset. For example, you could calculate correlation between every feature and the label on the training data, then choose the features that are the most strongly correlated with the labels
- Once you selected the best model, you could re-train it on the complete dataset, taking advantage of extra 30% that you initially selected as your test set
- **Statistical efficiency** of an algorithm is a rate at which its variance goes to 0 as the size of the dataset goes to infinity


## Learning theory

- Key assumption: all our data comes from some distribution D, all our samples are independent, there is a "true" value that we are trying to estimate
- We have a sample S (random variable) coming from D, that we put through some deterministic learning algorithm to produce hypothesis (or model) h(x): a function that makes predictions (also random variable)
- **Risk** or **Generalization error** of a model is an expectation `E[1{h(x)!=y}] under (x,y) ~ D`
- Basically, it's the ratio of wrong predictions on the real world data, an unknown quantity
- **Empirical risk** is an average `1/m*[sum of 1{h(xi)!=yi} over all i]`, given m samples
- Basically, it's the ratio of wrong predictions on the training set, can be calculated
- Let's call g the best possible hypothesis (overall)
- Let's call h* the best hypothesis in the given class of estimators (g can be completely outside of that class)
- Let's call h^ the hypothesis that we came up with during learning
- Empirical risk of h^ is going to be smaller than empirical risk of h* and that's OK! We are really interested in minimizing the generalization error, not the empirical risk
- `[Risk of g]` is the **irreducible error** ("Bayes error")
- `[Risk of h*] - [risk of g]` is an **approximation error**, comes from selected class of an algorithm
- `[Risk of h^] - [risk of h*]` is an **estimation error**, comes from data
- So, `[risk of h^]` is `sum of estimation error + approximation error + irreducible error`
- Estimation error has 2 parts: estimation bias and estimation variance
- Estimation variance is what is generally called "variance" in ML
- Estimation bias + approximation error is what is generally called "bias" in ML
- It is easy to see that by reducing the hypothesis space (by changing the class of an algorithm) you might be reducing the variance, but potentially moving away from g, thus increasing bias
- The learning algorithm can be anything, but there is a special class of algorithms: **Empirical risk minimizers (ERM)**
- Unsurprisingly, ERMs are looking for h^(x) that minimize empirical risk, i.e. minimize the training loss
- But of course, we are really interested in minimizing the generalization error
- Luckily, it can be proven that you can bound the difference between the empirical risk of h^ and the generalization error of h* to some arbitrarily small value gamma (margin of error), by increasing the sample size
- ERMs and MLEs are related under the hood


## Linear regression

- Linear regression attempts to map the input vector (features) to the output vector with a relatively simple linear model: `y = mx + b` (predict y given x)
- Or, in case of multiple features, `y = m1*x1 + m2*x2 + ... + mn*xn + b`
- Training is the process of determining what coefficients (parameters) will most closely match your training set. Training will attempt to minimize the overall error of your training set elements
- Error (in prediction) can be calculated using multiple formulas. Since y is what we are trying to predict, we are going to express error in terms of y
- **Mean Absolute Error** is the sum of all `|y - y'|` divided by number of data points. The disadvantage of this error function is that it is not continuous at the minimum, which is bad for gradient descent
- **Mean Square Error** is the sum of all squares of `(y - y')` divided by number of data points. This error function produces nice continuous function with an easy derivative
- We use this **error function** (or **cost function**) to guide the adjustments in coefficients, using, for example, the gradient descent
- We could also find the weights directly, by solving the system of equasions, but if we go in higher dimensions, this gets too complicated too quickly. Gradient descent turns out to be more practical
- **Batch gradient descent** computes the gradient for the whole data set. Depending on the size of the dataset, this might require scanning through the terabytes of data
- **Stochastic gradient descent** computes the gradient (and takes a step) for only one point at a time
- We moderate the adjustments by a **learning rate**
- Too big value of a learning rate can result in overshooting, too small one can result in very slow convergence
- As a rule of thumb, start with learning rate 0.01 and then find the one that works the best, trying different numbers on exponential scale
- Linear regression only works if the data is linear and is very sensitive to outliers; outliers make the line tilt
- If the data is not linear, we could use polynomial regression, where instead of the line we would use a polynomial, for example, `y = w1*x^2 + w2*x + w3`
- The more coefficients we use, the more complex the model becomes and the more chances are to overfit
- **Regularization** is penalizing the model complexity by taking it into account when calculating error
- In case of regression, the complexity of the model is "measured" by number and magnitude of the coefficients
- **L1 regularization** adds the sum of absolute values of the coefficients to the error
- **L2 regularization** adds the sum of squared values of the coefficients to the error
- L1 regularization can yield sparse models (i.e. models with few coefficients); Some coefficients can become zero and eliminated. Lasso regression uses this method
- L2 regularization will not yield sparse models and all coefficients are shrunk by the same factor (none are eliminated). Ridge regression and SVMs use this method
- We can adjust regularization by lambda to favor a simple model (large lambda) or a complex model (small lambda)
- Alternatively, consider **Locally Weighted Regression**: a method that performs a regression around a point of interest using only training data that are "local" to that point

### Probabilistic interpretation

- When we choose certain values of w and b, we essentially make an assumption that, for any sample, `y = w*x + b + e`
- In other words, we believe that the correct, true model is `y = w*x + b`, and if some y do not land on the line exactly, its due to an error e
- We will assume that this error is due to many different random influences (unmodeled effects and random noise)
- More formal way to say this: the value of e comes from i.i.d random variable E that is normally distributed with mean 0 and variance v
- By the Central Limit Theorem, and oversimplifying it, the combined impact of multiple random influences is normally distributed, even if the sources of those influences are not normally distributed
- So our assumption is reasonable from the theoretical standpoint, and it also holds (mostly) in the real world
- In practice, it just means that we usually use Gaussian distribution to model the noise
- So our model is `Y = w*X + b + E`
- Meaning Y is a random variable that, for each value x of random variable X, is normally distributed with a mean of `w*x + b` and variance v (i.e. all the variance is due to error)
- Formally, `P(Y=y|X=x;theta) ~ N(w*x + b, v)` where theta is a pair w, b
- **Likelihood** is the probability of Y taking a certain value y (in our case, conditioned on X), given certain values, w and b
- It is the same expression `P(Y=y|X=x;theta)`, but seen as a function of theta (and not the data)
- Likelihood expresses how likely it is to see a specific value of Y for a given theta, under our assumptions and keeping the data fixed
- If all the training data happens to fall under the line `y = w*x + b`, it's certainly possible that it is still a correct line and the correct model, and all the samples from our training set just happened to be registered with negative error, but it's definitely not the most probable scenario
- Imagine we have a coin that we believe is fair. Then we throw this coin 10 times and get 7 tails. Now, the question is: "just how likely would it really be to get 7 tails if the coin was indeed fair?"
- The principle of **Maximum likelihood estimation (MLE)** suggests choosing values of w and b so that the data looks as probable as possible
- In the example with coin it means changing our initial assumption and agree that the coin is, most likely, biased towards tails
- Likelihood L is maximum at the same point when log(L) is maximum. So instead of maximizing likelihood it is usually more convenient to maximize log likelihood
- It can be mathematically proven that maximizing likelihood is exactly the same as minimizing sum of squares of errors (when the model is assumed to be Gaussian)
- This justifies the choice of sum of squares as an error function for linear regression (and the validity of the whole model in theoretical sense)
- The whole method of linear regression is basically derived from these initial assumptions


## Logistic regression

- The same approach can be used for classification, in this case the line is going to divide the space into 2 categories (classes)
- Suppose we have 2 features (dimensions) x1 and x2 and we are trying to classify the points into 2 possible categories
- For example, we are trying to classify insects into caterpillars and ladybugs, and x1 is length and x2 is a width
- We will try to predict the class using the linear model `0 = w1*x1 + w2*x2 + b`
- In this case, w is weight, b is bias
- We have to come up with the initial values for weights somehow, can be random
- We plug features as values for x, and current weights as values for w, and get some number z as a result
- Then we pass z through a **sigmoid**: S-shaped logistic function with an y-intercept at 0.5
- Sigmoid allows us to interpret the result as a probability of something to be "true" or "false"
- Sigmoid produces 0.5 when z is 0, which gives our line an interpretation of a decision boundary: any point that sits on the line has an equal probability of belonging to any of two classes
- The number z we predicted expresses how much we would need to change values of x1 and x2 (in terms of w1 and w2) to get to the line, so it is a good measure of the distance from the decision boundary
- Although it is not a euclidian distance, but rather a manhattan distance
- Sigmoid approaches 1 and -1 at larger values of z, which matches our interpretation: the bigger the z is, more certain we are about the the prediction
- If we have n features, we are going to use a n-1-dimensional plane to separate points instead of a line. In this case the model will look like this: `0 = w1*x1 + w2*x2 + ... + wn*xn + b`
- Or, using vectors, simply `0 = W.T * X + b`
- Our goal when training the model is to come up with optimal values of weights W
- As usually, we can come up with an error function, use this function to drive the adjustments in weights (using gradient descent), moderate the adjustments by a learning rate, and re-evaluate the model performance until we are satisfied, then test our model using a testing set
- Because our function is non-linear (we use sigmoid), we cannot use mean square error as an error function
- We are going to stick to the probabilistic interpretation and maximize the likelihood
- It can be proven that maximizing the likelihood is the same as minimizing the **cross entropy**, so we will use cross entropy as an error function
- Cross-entropy measures the difference between 2 distributions (we want our output distribution to look like label distribution)
- There is no closed form solution, but instead of gradient descent you could use Newton's method, which is more efficient (converges quadratically), although it can become very costly in high dimensions
- Similar to linear regression, if the data is not linearly separable, we could fit our data with a higher degree polynomial (e.g. quadratic function)
- Ideally, we would want equal distribution of classes in the training dataset; if this is not the case (**class imbalance**), you could modify the model to weight model error by class weight when fitting the coefficients
- sklearn LogisticRegression has a parameter class_weight; when set to 'balanced', it adjusts weights inversely proportional to class frequencies in the input data
- We are not forced to use 0.5 as a treshold after applying sigmoid. We could move it up and down to favor precision or recall. We could use **ROC curve** to visualize the results (see appendix for details)
- When number of features greatly exceeds the dataset size, the logistic regression tend to overfit badly, which can be combatted by adding regularization
- If, instead of MLE approach, you decide to go bayesian way and do maximum aposterory estimation (MAP), it can be shown that the prior distribution acts as a regularizer. So doing MLE with regularization is kind of "going bayesian"


## Generalized Linear Models

- Turns out both linear regression and logistic regression are special cases of a broader family of models, called **Generalized Linear Models (GLMs)**
- GLMs goal is, given data X, to predict the expected value of sufficient statistic T(Y) (for most cases, for example, linear regression, T(Y) simply means Y)
- GLMs assume that given X and theta, `[Y|X;theta]` is going to be a random variable that follows some exponential family distribution with some **natural parameter** eta
- To understand this, remember how for linear regression we assumed that given any particular value of x, Y was a random variable normally distributed with mean `w*x` (natural parameter for gaussian is mean)
- And most of the commonly used distributions form an exponential family
- As a design choice, we will assume that eta and X are related linearly through theta (just like w*x in linear regression)
- One of the most important decisions to make is to pick a correct distribution to model the target variable
- You cannot simply collect the data and throw it to a linear regression, you have to think what you are doing
- For example, if you are predicting the number of customer visits per hour for a website, you should use Poisson distribution
- Poisson distribution expresses the probability of a given number of events occurring in a fixed interval of time, cannot be negative
- When mean of Poisson is close to zero, it is right skewed
- The choice of distribution allows us to construct an appropriate model
- For a normally distributed (Gaussian) random variable, we construct linear regression, with simple `w*x`, and the least squares as an error function
- For Bernoulli distribution, we construct logistic regression with `sigmoid(w*x)`
- For multinomial distribution (k labels), we construct softmax regression, with `softmax(w*x)`
- For Poisson distribution, we construct Poisson regression (https://en.wikipedia.org/wiki/Poisson_regression), turns out, also uses least squares as an error function
- While sigmoid and softmax are kind of convenient functions we might have just invented anyways, they can actually be mathematically derived from parameters of exponential family distributions
- Since all GLMs assume probabilistic interpretation, our goal when training the model is always the same: maximizing the likelihood
- The expressions for likelihood and error function are also derived in the similar way

### Recipe

- Assume a certain distribution for `[Y|X;theta]`, pick an appropriate one (Gaussian for real numbers, Bernoulli for binary outcomes, Poisson for counters etc.)
- Use the exponential form of the distribution and relate its natural parameter eta to X through theta
- Formulate you hypothesis `h(x;theta) = E[T(Y)|X;theta]`
- For simple models, `h(x;theta)` is simply `E[Y|X;theta]`, but maybe you don't want to spit out Y directly and return some T(Y) instead (like in a softmax)
- `h(x;theta)` gives you the output, whatever you train your model to return
- Since you know the distribution, you know how to find the expected value, and you can exress that in eta
- Express eta through X and theta, you get the model; now you just need to train it, i.e. find the good theta
- Find `p(y|x;theta)`: the PDF of Y under theta, conditioned by X
- Likelihood is `P(Y=y|X=x;theta)` as a function of theta
- Maximize this function in respect to theta (on your dataset), usually by maximizing the log of it
- The good news: the gradient descent is always going to be the same: `theta_j = theta_j + alpha*(y_i - h(x_i))*x_i_j`
- Use the argmax theta as a parameter of your model
- Done


## Support vector machines (SVM)

- Built on the same principle as logistic regression, but tries to find the model that not only classifies the best, but also maximizes the distance between the points and the line
- Formally, we want `w1*x1 + w2*x2 + b >> 0` in the case y > 0 and `w1*x1 + w2*x2 + b << 0` in the case y = 0
- Consequently, if we pass this value to the sigmoid, we want to see values close to 1 and 0 and not something like 0.49 and 0.51
- In plain English, we want the model to be very confident about its predictions
- We hope to achieve that by finding the optimal position and angle of the line (or hyperplane)
- But notice that it would be very easy to achieve our formal requirement by simply multiplying W and b by some arbitrarily large number, but this would be useless, because the line would look exactly the same
- So we need to restrict magnitude of W and b in some way to force it to "wiggle" instead (during optimization)
- For that, we are going to opt for such a scale of W so that the points closest to the decision line (on both sides) would land exactly on the margins defined by 2 lines: `w1*x1 + w2*x2 + b = 1` and `w1*x1 + w2*x2 + b = -1`
- We will call the points that lay exactly on 2 margins **support vectors**
- If we choose to use values -1 and 1 to designate negative and positive samples, our condition can be re-written more concisely as `y(w1*x1 + w2*x2 + b) >= 1` for all datapoints, reading as "all points are correctly classified and at least a margin away from the decision boundary"
- Now we will try to find such a values for W and b so that the geometrical distance between 2 margins is the largest, while still satisfying the condition ("the widest street approach")
- Finding those values for W and b is a tricky optimization problem (see appendix for details), but once you solve it, you discover that your solution depends only on a dot product of the feature vectors [x1, x2], and ONLY the feature vectors of data samples that are support vectors
- The decision rule also ends up being dependent only on inner products of dataset features vectors with the unknown point feature vectors
- This is also a key to the whole method, as this gives a very efficient and elegant algorithm for the optimization, even though it is somewhat difficult to understand completely
- But all you really need to know in practice, is that SVM is a **Maximum Margin Classifier**
- In our description of SVMs so far, we assumed that the data is actually linearly separable
- Sometimes, of course, it's impossible to separate points on a 2-dimensional plane using a simple line
- However, if we send the points in higher dimensions, often it suddenly becomes very easy to separate them using n-1-dimensional plane
- We could send a point with coordinates (x, y) into 5-dimensional space by using "artificially made" coordinates `(x, y, x^2, x*y, y^2)`. We could even send it to higher dimensions, using coordinates `(x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3)` etc.
- By doing this, the 2-dimensional plane will curve, and the points will suddenly move to the different hights in a 3rd (and higher) dimension
- At worst, we could send points into infinite dimensions, which is done btw :)
- The only nuisance is that with the number of features increasing, computing dot product of X becomes a very heavy operation (especially in case of infinite dimensions :D)
- Luckily, there exist **kernel functions** that allow us to compute the dot product very efficiently using some shortcuts and without actually doing vector multiplication. They can handle infinite dimensions (magic of math)
- There are tons of ways to send points to higher dimensions with corresponding kernels (e.g. RBF kernel)
- SVMs are among the best "off-the-shelf" supervised learning algorithms, their implementations are very robust and they "just work". Some even argue that they are much better than neural networks (e.g. their inventor)


## Perceptron

- Very similar to logistic regression, except instead of a sigmoid, we use a step function that outputs only 0 or 1
- More precisely, step function produces 0 for z < 0; otherwise, 1
- It is difficult to endow the perceptron's predictions with meaningful probabilistic interpretations, or derive the perceptron as a maximum likelihood estimation algorithm
- So (my understanding) it does not belong to a family of Generalized Linear Models
- If the data is not linearly separable, it can always be classified using 2 or more lines
- This leads to an idea of using 2 (or more) peceptrons to train 2 different linear models and then using a third perceptron to combine the results
- This is a fundamental building block of a neural network


## k-Nearest Neighbors

- Can be used for classification or regression
- Remember that linear regression uses model `y = mx + b`; once it learns the correct values for m and b, it never looks at the original dataset again
- k-Nearest Neighbors instead uses data-centric approach: for any value of x it finds k data points that are closest to x and uses them to predict y
- A commonly used distance metric for continuous variables is **Euclidean distance**
- For discrete variables, you can use different metrics, e.g. **Hamming distance**
- For regression, the value is the average (mean) of the values of k nearest neighbors
- For classification, an object is classified by a plurality vote of its neighbors
- A useful technique can be to assign weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones
- A common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor
- If the class distribution is skewed (class imbalance), the more frequent class tend to dominate the prediction of the new example (they tend to be common among the k nearest neighbors due to their large number)
- As with other methods, we could weight the distances proportional to class frequencies
- The best choice of k depends upon the data
- Generally, larger values of k reduces effect of the noise on the classification, but make boundaries between classes less distinct
- This approach is memory-inefficient
- On the flip side, you can improve your model at any time by simply adding more data points


## Naive Bayes

- This is one of the **Generative learning algorithms**, as opposed to **Discriminative learning algorithms** (e.g. logistic regression)
- The big difference is that in discriminative learning algorithms we are trying to maximize the `likelihood L(y|x, theta)`, while in generative learning algorithms we are trying to maximize the `likelihood L(x,y;theta)`
- We know probabilities of certain events and we are inferring the probability of other events, based on known probabilities
- For example, we know the probability of email being `spam = P(spam)`
- We also know the probability of seeing the word "money" in spam emails = `P("money" | spam)`
- We also know the probability of seeing the word "money" in all emails, spam or `ham = P("money")`
- All these probabilities can be calculated simply looking at the training data (the MLE estimator for these probabilities will be just frequencies)
- So now using a simple formula, we can calculate the probability of email being spam given that it contains the word "money": `P(spam | "money") = P("money" | spam) * P(spam) / P("money")`
- To extend this to multiple words let's say our mail contains a subject "easy money", then we will calculate `P(spam | "money" & "easy")`
- If seeing words "money" and "easy" in a spam email were 2 conditionally independent events, `P("money" & "easy" | spam)` could be calculated by simply multiplying probabilities of individual events: `P("money" | spam) * P("easy" | spam)`
- Conditional independence means that if I tell you that a particular email is spam, the knowledge of whether word "easy" appears in the message will have no effect on your beliefs about whether word "money" appears in that message
- Usually the events are dependent, so unfortunately we cannot apply this simple formula... Or can we?
- We will use a naive approach and simply pretend all the events are conditionally independent. In practice it does not seem to matter that much, but makes calculation super easy
- But this is why this method is called "Naive Bayes"
- So the probability will be `P(spam | "money" & "easy") = P("money" & "easy" | spam) * P(spam) / P("money" & "easy")`
- Which we will naively convert to `P(spam | "money" & "easy") = P("money" | spam) * P ("easy" | spam) * P(spam) / P("money" & "easy")`
- `P("money" & "easy")` cannot be calculated by simply multiplying individual probabilities, because we don't assume general, but only conditional independence
- It is possible to express `P("money" & "easy")` in terms of known probabilities, but it's just a normalizing factor (and a constant, for a given dataset), so we just drop it
- So technically speaking we are not calculating the probability, but some value that is proportional to probability. However, it's enough to calculate the proportion `P(spam | "money" & "easy")/P(not spam | "money" & "easy")`
- This method works best if we need a binary classificator, but you can apply it to multiple outcomes
- And if you have a continuous outcome, you can discretize it into buckets (usually 10 works well)
- Computationally this method is very efficient, it's only counting and multiplying, quick to train, easy to implement, but is not very accurate and usually loses to logistic regression and more sophisticated algorithms
- You should be careful modeling probability of words you have never seen before as 0. For example, if you never saw word "stanford" in spam emails, it doesn't mean the probability `P("stanford" | spam) = 0`
- If you do take `P("stanford" | spam) = 0`, you will end up multiplying and dividing by 0 everywhere
- **Laplace smoothing** suggests "adding one" to all counts. This leads to estimating `P(spam)` as:
```
	(# of spam emails) + 1 / ((# of spam emails + 1) (# of non-spam emails + 1))
```
- Pre-processing data with PCA eliminates correlated features, so it might help with satisfying "naive" assumption. But you need to remember that:
	a) Correlation of 0 does not imply independence
	b) Naive bayes requires conditional independence, not just independence


## Gaussian Discriminant Analysis (GDA)

- Another generative learning algorithm
- GDA tries to fit a Gaussian distribution to every class of the data, separately
- We assume `p(x|y)` is a multivariate normal distribution in n dimensions
- We assume `p(y)` is Bernoulli
- X is features of the dataset, e.g. weight, height etc., have to be real numbers
- Y is a label of the class, e.g. 0 for "cat" and 1 for "dog", so naturally it is Bernoulli
- `p(x|Y=0)` is the PDF for cat's "features", `p(x|Y=1)` is the PDF for "dog's features" (both multivariate normal)
- `p(y)` is the class prior, the PDF for Bernoulli on labels
- We just write their standard, well-known PDFs, `p(x|y)` parametrized by vector of mu's and the covariance matrix, `p(y)` parametrized by p (vector of mu's and covariance matrix is just how the multivariate normal distribution is parametrized)
- Usually people use the same covariance matrix for both distributions, for simplicity, but you could use 2 different ones
- Then we express the joint probability `P(X=x,Y=y)`, which will give us the expression for the likelihood
- That done, we will fit the distributions to the data, using maximum likelihood principle; this will give us the values for all the mu's, sigma^2's and p
- To do that, we take log of the likelihood, find a partial derivative with respect to each parameter, equal to 0 and find the expression for the parameter
- The expressions for parameters that we derive this way turn out to be very unsurprising, e.g. p is a fraction of y = 1, mu is an average etc.
- Now we can now use Bayes rule to derive the posterior, `p(y|x)`, i.e. guess whether we see a cat or a dog
- As usual, `p(y|x) = p(x|y)*p(y)/p(x)`
- We could calculate `p(x) = p(x|Y=1)*P(Y=1) + p(x|Y=0)*P(y=0)`
- But sice we are just looking for argmax of `p(y|x)` in respect of y (i.e. which label would give the highest score), we don't need to normalize by `p(x)`
- GDA has stronger assumptions than linear regression, and if those assumptions turn out to be correct, performs better, especially on smaller datasets
- This is a theme in ML, when you don't have too much data, you need to compensate by building more knowledge into the model in the form of assumptions


## Decision trees

- Non-linear model, instead of producing a decision line, tries to partition the whole sample space into regions, in a greedy, top-down, recursive manner
- What this produces is basically a huge pyramid of if-else statements, each making a question about a single feature: whether it is above or below a certain treshold
- Every time we answer yes or no, we split our dataset into 2 subsets, and then repeat
- But because it is machine learning and not just programming, we don't code it by hand. Instead, we rely on the algorithm to build the tree for us
- The trickiest part of an algorithm is to decide on which feature to split every time we split
- We are going to use a greedy approach where on each step we are going to maximize the information gain
- Information gain is calculated based on entropy
- Entropy is a measurement of uncertainty. The set with all the same items has an entropy of 0 (whatever item we pick, we know what it is going to be beforehand). The set with items of 2 types split 50/50 has an entropy of 1 (the highest level of uncertainty we can obtain with only 2 types of items). There is a slightly complicated formula that allows to calculate the entropy
- Information gain is the difference between an entropy of the parent and the average entropy of the children
- So we are going to try every feature and see how much information gain we can get, and every time split on the feature that gives the maximum information gain on that step
- It is very important to decide when to stop splitting. If we reach the point when every child has a pure subset (0 entropy), we are most likely to overfit
- In the most extreme case, we will end up with a single datapoint in each region
- A number of hyperparameters can be used to combat overfitting: maximum depth, minimum number of samples per leaf, minimum number of samples per split, maximum number of features
- Instead of stopping early, you could let your tree grow to the end, and then prune it (remove some leaves)
- These methods are considered regularization
- Random forests are made of many decision trees that are built from the randomly chosen subsets of the same dataset (bagging) and randomly chosen features (feature bagging). The final decision is made by a majority vote
- With **bagging**, you are using your sample as a population and you are drawing smaller samples from it (with replacement). It can be shown mathematically that this decreases the variance of the predictions
- Random forests reduce overfitting and the negative impact of the greediness. However, they are more difficult to interpret and more expensive to build
- Decision trees are simple to understand and to interpret, they can even be visualised
- Decision trees are pretty fast, require little data preparation, are able to handle both numerical and categorical data

### Boosting

- The idea of **boosting** is that, after building a tree, you look at the misclassified examples and give them more weight
- You then re-build the tree again, this time getting better boundaries


## Data preparation

https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-data
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
https://cs231n.github.io/neural-networks-2/
https://keras.io/preprocessing/image/

- Check that all data is numeric. Everything has to be converted to numbers
- Identify categorical features. Many algorithms cannot work with categorical data (e.g. "book genre" or "author id"), since they rely on the notion of distance. Simple solution is one-hot encoding
- Check for missing values. Missing values can imputed with the mean. Another option is to simply drop rows with missing values
- Check for outliers. Outliers should be analyzed and optionally removed
- Zero-center the data. Can be done subtracting the mean across every individual feature (`X -= np.mean(X, axis = 0)`)
- Normalize the data to make sure every dimension is of approximately the same scale. Data is often scaled in the [0, 1] range or [-1, 1] range. Some scaling methods are very sensitive to the presence of outliers
- One way to scale data is to divide each dimension by its standard deviation (`X /= np.std(X, axis = 0)`). Another popular (and very similar) way is to apply min-max
- Labels for binary classification should be converted to 0 and 1
- Check for correlated features. Correlated features affect the output in the similar way, so they should be eliminated. Highly correlated features are often the same feature just expressed in different units, or a derived value. Keeping the same feature twice would result in that feature dominating the prediction
- Remove features with low variance. If the variance is zero, it means that the feature is constant. If a feature is constant, then it cannot be used for finding any interesting patterns
- Instead of simply dropping correlated features or low-variance features, you could use dimensionality reduction (PCA), that would discover latent features and retain more useful information
- Also consider dimensionality reduction if there are too many features. This can help in reducing model variance and speeding up the algorithm
- In particular, one-hot encoding can lead to explosion of features (curse of dimensionality), so it is often followed by dimensionality reduction step (PCA)
- Verify class distribution. Ideally, you want every class to be equally represented in the training set. In case of images, you could generate new images for underrepresented class by deforming existing ones (or adding some noise to existing ones)
- Any preprocessing statistics must only be computed on the training data, and then applied to the validation/test data. E.g. computing the mean and subtracting it from every sample across the entire dataset and then splitting the data into validation/test would be a mistake!


## Debugging ML algorithms

- Start with some "quick and dirty" algorithm
- When algorithm is not performing well enough, suspect either high bias or high variance
- Compare train error with test error to see which one it is
- One good way to visualize this is by plotting the error as a function of training set size
- High variance: test error is still decreasing a set size increases, but there is a large gap between training and test error
- High bias: training error is unacceptably high, small gap between training and test error
- Apply solutions that target either high bias or high variance (see above)
- For example, if your training error is already bigger than your target, no point in collecting more data
- When some algorithm performs better that yours (in terms of precision/recall), suspect that you are either not converging, or you optimizing the wrong function
- Compare your loss with the loss of another algorithm (or human)
- If your error is bigger than the one of another algorithm, you are not converging
- If your error is smaller than the one of another algorithm, you are actually training well, but simply optimizing the wrong function
- Choosing a good cost function can be a very hard problem (think how would you evaluate how well the helicopter made a turn, what does it mean for a helicopter to make a good turn?)
- Once you improve one thing, you need to re-evaluate the performance and see what is the next best thing to work on
- Often you have a complex pipeline of algorithms performing a multi-step task. The challenge in this case is to correctly attribute the fraction of the final error to each of individual components. You can do that by plugging in, instead of each component, the manually done perfect implementation and see how this affects the final result


## ROC Curve

- When classifying the points, we could move theshold up and down
- Moving it down, we would classify more points as "Yes", catching more true positives, but also increasing false positives, so moving it towards high recall but a low precision
- Moving it up, we would classify more points as "No", avoiding false positives, but also dropping in true positives, so moving it towards low recall but a high precision
- So unavoidably, we the only way to increase the true positive rate is to also accept increase in false positives
- If we plot number of tp and fp, weighted, `(tp / all positives) to (fp / all negatives)`, we can see how exactly the number of fp grow with growth of tp with the respect to theshold
- Every point on that line will correspond to a certain treshold
- When we classify randomly (50/50), the dependency is linear: more tp - more fp. Basically, whatever treshold we pick, we always get the exact ratios
- But when we have a well trained model and well-sepatated data points, the chart tends to look more like a an "r": Initially, lowering the theshold, we are able to catch more and more tp without suffering from increase in fp. But eventually we lower it too much and number of fp begins to grow very fast
- The better the model and the better the separation, the more the chart is curved, meaning we can lower the theshold so to catch almost all tp before seeing a sudden increase in fp
- This can be quantified by taking the square under the curve. The lowest possible value is 0.5, the highest is 1
- So ROC curve can be used to compare different models (e.g. logistic regression vs random forest)


## SVM optimization

- We will call the distance from a data point to the decision boundary, `gamma = y(W*X + b)/||W||`, the "geometric margin", where `||W||` is the norm of W and X is a feature vector
- The expression "closest point to the line", mathematically, translates into the `[min of all geometric margins over the whole dataset]`, and we want to maximize that
- So we want `y(W*X + b)/||W||] >= [min gamma over dataset]`, over the whole dataset, with [min gamma over dataset] being as large as possible
- If, by our design choice, we fix [min gamma over dataset] to be `1/||W||`, the whole task boils down to maximizing `1/||W|| subject to y(W*X + b) >= 1 for all datapoints`
- We will call the expression `y(W*X + b)` the "functional margin"
- It is mathematically more convenient to maximize `1/2*||W||^2` instead, so we will do that
- Lagrange multiplier technique: function `f(x,y)` subject to constraint `g(x,y)=c` has a maximum when the two functions touch, that is, when their gradients are parallel. Note that gradients do not have to be of equal length, they are linearly related through the constant alpha 
- So that is one of the points where `grad(f(x,y)) = alpha*grad(g(x,y))`
- As a note, to apply the technique, number of dimensions of f and g have to be equal
- Taking partial derivatives of f and g in respect of x and y gives 2 equasions `f'dx = g'dx` and `f'dy = g'dy`, the condition `g(x,y)=c` is the third, a perfectly solvable situation
- This system of equasions can actually have multiple solutions, and in that case you have to check all of them, to see which one actually maximizes f
- Turns out, we can also pack f and g into a single function, a Lagrangian, so that both the function we are trying to maximize and the constraint get incorporated into it: `L(x,y,alpha) = f(x,y) - alpha*(g(x,y) - c)`
- Turns out, finding the maximum of L is the same as finding the point where gradients of f and g are parallel
- Why it works: if you take gradient of this new function L, equate to 0, and then look at partial derivatives with respect to x, y and alpha, this will be exactly the same system of equasions as before, actually, quite beautiful
- In our case, `L(W,b,alpha) = 1/2*||W||^2 - [sum of alpha_i*(y_i(W*X_i + b) - 1)]`
- Note that we are actually completely ignoring any of the datapoints that are not support vectors
- So we take partial derivatives with respect to W and b, equal to 0, and this gives us 2 expressions that hold when L is maximized
- So we re-plug the new expressions into L and and, as a result, W and b disappear
- All we need now is to maximize the new L with respect of alpha (with 2 conditions that we found above), the mathematicians know how to do that, and luckily, it is a convex function, which means it has only one minimum
- What we discover is that this optimization problem depends, in terms of input, only on the dot product of feature vectors X
- And the only points that actually affect the solution are support vectors, which are very few
- So if you know the dot product, you don't even need the original feature vectors, and that is what kernel functions take advantage of
- Also, maximizing `||W||^2` can be proven to have a similar effect as adding L2 regularization term, so SVMs have regularization "built-in", which prevents overfitting, even in the infinite dimension feature space


## SVM some intuition about sending points into high-dimensional space

- The intersection of an n-1-dimentional plane that separates the points and the new multi-dimensional surface on which the points lie is a shape described by a high degree polynoms (polynomial kernel)
- For example, by sending point `(x,y)` into 3-dimensional space by using coordinates `(x, y, x^2 + y^2)` we are essentially using the distance from the center as a hight of the point. The surface made by this trasformation will look as a cone. The plane that classifies the points will separate lower points from the higher points, and the intersection with the cone will give the circle. The formula of the circle is `x^2 + y^2 = 0`, which is exactly what we used for the 3rd coordinate
- Another way to think of it is to imagine using a circle to classify the points


## Cross-entropy

- Self-information expresses the level of surprise associated with one particular outcome of a random variable
- When tossing a coin, you don't know whether it is going to be head or tail, so it is a surprise every time. When someone tells you the outcome, you get a lot of information
- When tossing the same coin, you know for sure it will fall on the ground, this is no surprise at all. When someone tells you it did fell on the floor, you say "duh"
- Entropy quantifies how "surprising" the entire random variable is, averaged on all its possible outcomes
- Cross-entropy compares 2 probability distributions
- Training set maximizes prediction for every data point, so the entropy is going to be very low
- Model predictions are between 0 and 1, more confident is the model, the smaller is the entropy


## Some python code

```
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cross_entropy(t, y):
    return -(t*np.log(y) + (1 - t)*np.log(1 - y))

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))
```