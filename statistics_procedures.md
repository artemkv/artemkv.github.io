# Statistics, elementary procedures
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## Statistical modeling

- **Statistical model** is the set of possible observations (sample space) and the associated probability distribution
- The end result of a statistical analysis is the estimation of parameters of a postulated model


## Statistical inference

- **Population** consists of the totality of the observations with which we are concerned
- **Real population** is one in which all potential observations are accessible at the time of sampling
- **Hypothetical population** is one in which all potential observations are not accessible at the time of sampling
- **Parameter** is any descriptive measure of a population such as the population mean
- In classical interpretation, parameters are numbers, not random variables
- **Sample** is any subset of observations from a population
- **Random sampling** occurs if all potential observations in the population have an equal chance of being included in the sample
- **Statistic** is a numerical measure calculated from sample data such as the sample mean
- All statistics are random variables
- **Sampling distribution** is the probability distribution of a statistic
- The purpose of statistical inference is to draw conclusions about population parameters from statistics
- With **survey**, we obtain a random sample from some real population
- With **experiment**, subjects are assigned randomly to one of two groups: a treatment group and a control group
- Before performing a statistical inference procedure, examine the sample data! If any of the conditions required for using the procedure appear to be violated, do not apply the procedure


## Sampling distribution of the mean and Central Limit Theorem

- **Sample mean** is the mean of the observations for a sample (`X bar`)
- The **sampling distribution of the mean** is the probability distribution of sample means (for all possible random samples of a given size)
- Therefore, the mean of the sampling distribution of the mean is the mean of all sample means (`mu of X bar`)
- The mean of all sample means is equal to the mean of the population (`mu of X bar = mu`)
- **Standard error of the mean** is the standard deviation of the sampling distribution of the mean (`sigma of X bar`)
- Standard error of the mean is a measure of the average amount by which sample means deviate from the mean of the sampling distribution (or the population mean)
- `Sigma of X bar = sigma / sqrt(n)`, where `n` is the sample size
- **Central Limit Theorem:** regardless of the population distribution, the shape of the sampling distribution of the mean approximates a normal curve if the sample size is sufficiently large
- In other words, the distribution of `Z = (X bar - mu) / sigma of X bar` is the standard normal distribution


## Hypothesis testing

- **Statistical hypothesis** is a statement about the nature of a population, often stated in terms of a population parameter (e.g. population mean)
- **Null hypothesis (`H0`)** is the "status quo hypothesis": nothing special is happening (there is no effect)
- **Alternative hypothesis (`H1`)** is the question to be answered or the theory to be tested ("a scientific discovery")
- To test the hypothesis, we take a random sample from the population and use the data contained in this sample to provide evidence that either supports or does not support the hypothesis
- Using the classic approach, we either:
	- reject `H0` in favor of `H1` if there is sufficient evidence in the data (**statistical significance**), or
	- fail to reject `H0` because of insufficient evidence in the data
- The retention of `H0` can not be interpreted as proving `H0` to be true
- The decision to retain `H0` implies not that `H0` is probably true, but only that `H0` could be true, whereas the decision to reject `H0` implies that `H0` is probably false
- **Decision rule** specifies precisely when `H0` should be rejected
- The decision is based on the rarity of the outcome obtained when taking a random sample
- **Common outcome** is one that can be readily attributed to chance, and therefore can be viewed as a probable outcome under the null hypothesis
- **Rare outcome** is one that can not be readily attributed to chance, and therefore cannot be reasonably viewed as a probable outcome under the null hypothesis
- Common outcome signifies a lack of evidence to reject `H0`
- **Level of significance** indicates the degree of rarity required of an observed outcome in order to reject `H0`; it is the probability of committing a type I error
- **Type I error**: rejection of the null hypothesis when it is true ("false alarm" or sending an innocent person to a jail)
- **Type II error**: failure to reject the null hypothesis when it is false ("miss" or letting a criminal walk away from the justice)
- The probability of committing a type I error is denoted as `alpha`
- The probability of committing a type II error is denoted as `beta`
- Usual practice is to control for `alpha`, `beta` being dependent on `alpha`
- This follows from the philosophy that the maximum risk of making a type I error should be controlled
- Unless there are obvious reasons for selecting either a larger or a smaller level of significance, use the customary 0.05 level
- The next best choice for `alpha` is 0.01
- The **power** of a test is the probability of rejecting `H0` given that `H1` is true: `1 - beta`
- Power of a test is a measure of the **sensitivity** of the test
- All other things being equal, using larger samples gives more information and thereby increases power
- **Effect** is any difference between a true and a hypothesized value of a population parameter
- An unduly small sample size will produce an insensitive hypothesis test that will miss even a very large, important effect
- However, an excessively large sample sizes can produce an extra-sensitive hypothesis test that detect even a very small, unimportant effect
- You should never look at your sample data when making a statistical hypothesis, you should formulate your hypothesis before collecting the sample!


### P-Value

- **Critical region** (aka rejection region) is a set of values for the test statistic for which the null hypothesis is rejected
- **Critical value** is the last value that we observe in passing into the critical region
- Using level of significance does not account for values of test statistics that are close to the critical region; very similar values can lead to two opposite conclusions
- At the same time, it treats extreme values of a statistic in the same way as the values just beyond the critical value
- Instead of reporting the decision of retaining or rejecting the null hypothesis, we could report the degree of rarity of the observed value, using p-value
- **p-value** indicates the degree of rarity of the observed test result + all potentially more deviant test results
- Hint: it's area under the curve beyond the value of the observed test statistic
- Formally, for the given sample, p-value indicates the smallest `alpha` at which we would reject `H0`
- **Golden rule**: reject `H0` at level `alpha` when `p-value <= alpha`


## Estimation (confidence intervals)

- **Statistical estimation** is a form of statistical inference in which we use the sample data to determine an estimate of some population parameter
- You make an estimation using an **estimator**: a rule for calculating an estimate of a given quantity based on observed data
- A rule can be anything, some formula or even a constant
- Estimators can be evaluated and compared based on **bias** and **variance**
- **Bias** of an estimator is the difference between this estimator's expected value and the true value of the parameter being estimated
- **Variance** of an estimator indicates how far, on average, the collection of estimates are from the expected value of the estimates
- **The most efficient estimator** is the one with no bias and the smallest variance
- **Point estimate** uses a single value to represent the unknown population parameter, usually the value of a statistic
- Point estimates convey no information about the degree of inaccuracy due to sampling variability
- **Confidence interval** uses a range of values that, with a known degree of confidence (certainty), includes the unknown population parameter
- **The degree of confidence** (`1 − alpha`) indicates the percent of time that a series of confidence intervals includes the unknown population parameter
- 95% and 99% confidence intervals are the most prevalent
- The greater the confidence level, the wider the confidence interval
- The larger the sample size, the narrower the confidence interval
- If width and confidence level are given, then we must determine the minimal sample size needed to meet those specifications


## Methods

- Methods for population mean, one or two populations, for independent or paired samples
- Methods for population proportion, one or two populations
- **Population proportion** is the proportion (percentage) of a population that has a specified attribute
- Methods for categorical data
- You must check for conditions to make sure the method is valid
- You may use quantile-quantile plot (Q-Q plot) to validate the normality assumption (if normal, should look like a single straight line)

### Catalog

- Hypothesis test for one population mean
	- Sigma is known
		- z-Test
	- Sigma is unknown
		- t-Test
- Hypothesis test to compare two population means
	- Sigma is known
	- Sigma is unknown, assumed equal
		- Pooled t-Test
	- Sigma is unknown, not assumed equal
		- (Welch) Non-pooled t-Test
	- Paired sample
		- Paired t-Test
- Hypothesis test for a population proportion
	- One-Proportion z-Test
- Hypothesis test to compare two population proportion
	- Two-Proportions z-Test
- Estimation for one population mean
	- Sigma is known
		- z-Interval procedure
	- Sigma is unknown
		- t-Interval procedure
- Estimation for the difference between two population means
	- Sigma is known
	- Sigma is unknown, assumed equal
		- Pooled t-Interval procedure
	- Sigma is unknown, not assumed equal
		- (Welch) Non-pooled t-Interval procedure
	- Paired sample
		- Paired t-Interval procedure
- Estimation for a population proportion
	- One-Proportion z-Interval Procedure
- Estimation for the difference between two population proportions
	- Two-Proportions z-Interval Procedure
- Hypothesis test to compare k population means
	- One-Way ANOVA Test
- Categorical data
	- Compare observed frequencies to expected frequencies
		- Chi-Square Goodness-of-Fit Test
	- Decide whether two variables are associated
		- Chi-Square Independence Test
	- Compare the distributions of a variable of two or more populations
		- Chi-Square Homogeneity Test

### Hypothesis test for one population mean when sigma is known (z-test)

#### Assumptions

- Population is normally distributed (or not too far)
- Population standard deviation sigma is known
- The sample size is sufficiently large (>30)

#### Steps

- `H0` is: population mean `mu` is some number `mu0`
- `H1` is: `mu != mu0` (two-tailed), `mu < mu0` (left tailed), or `mu > mu0` (right tailed)
- Decide on significance level `alpha`
- Compute test statistic `z = (X bar - mu0)/(sigma/sqrt(n))`
- `n` is sample size
- Find critical values: `+/- zalpha/2` for two-tailed test, `-zalpha` for left tailed test, or `+zalpha` for right tailed test
- Reject `H0` if `z` falls in the rejection region; otherwise, do not reject `H0`
- Or, calculate P-Value and reject if `P <= alpha`

### Estimation of population mean when sigma is known (z-interval procedure)

#### Assumptions

- Population is normally distributed (or not too far)
- Population standard deviation sigma is known
- The sample size is sufficiently large (>30)

#### Steps

- For a confidence level of `1-alpha`, find `zalpha/2`
- Construct the interval `X bar +/- zalpha/2*(sigma/sqrt(n))`
- `n` is sample size
- `X bar` is the sample mean

### Hypothesis test for one population mean when sigma is unknown (t-test)

#### Assumptions

- Population is normally distributed (or not too far)
- The sample size is sufficiently large (>30)

#### Steps

- `H0` is: population mean `mu` is some number `mu0`
- `H1` is: `mu != mu0` (two-tailed), `mu < mu0` (left tailed), or `mu > mu0` (right tailed)
- Decide on significance level `alpha`
- Compute test statistic `t = (X bar - mu0)/(S/sqrt(n))`
- `n` is sample size
- `S` is the sample standard deviation
- Find critical values: `+/- talpha/2` for two-tailed test, `-talpha` for left tailed test, or `+talpha` for right tailed test, with `df = n−1`
- `df` is "degrees of freedom"
- Reject `H0` if `z` falls in the rejection region; otherwise, do not reject `H0`
- Or, calculate P-Value and reject if `P <= alpha`

### Estimation of population mean when sigma is unknown (t-interval procedure)

#### Assumptions

- Population is normally distributed (or not too far)
- The sample size is sufficiently large (>30)

#### Steps

- For a confidence level of `1-alpha`, find `talpha/2` with `df = n−1`
- Construct the interval `X bar +/- talpha/2*(S/sqrt(n))`
- `n` is sample size
- `df` is "degrees of freedom"
- `X bar` is the sample mean
- `S` is the sample standard deviation
