# Probability, core modules
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Introduction to Probability, Statistics, and Random Processes by Hossein Pishro-Nik](https://www.probabilitycourse.com/)
- [MIT 6.041 Probabilistic Systems Analysis and Applied Probability](https://www.youtube.com/playlist?list=PLUl4u3cNGP61MdtwGTqZA0MreSaDybji8)
- [Harvard Statistics 110: Probability](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)


## My personal takeaways

- While math is really clear-cut, the usefulness of a model greatly depends on your assumptions about the world, and which assumptions you bring into your model
- Our daily life does not provide us with enough intuition for the probabilistic modeling, everything needs to be reasoned very carefully and step by step. There are many examples of fallacies that are just mind-blowing
- Brute-force is to enumerate every outcome possible and pick only the ones we like, then work with individual probabilities of outcomes. Ignore all the outcomes that are not interesting
- Individual probabilities are input into the model, must be known, cannot be assumed!
- Test for potential causality between events by testing for independence, if you believe they are causally related, use Bayes rule for inference
- If dependent, measure the strength of the relationship using correlation
- Probability for events, PDF/PMF/CDF for random variables!
- Law of large numbers is about value of the sample mean, the CLT is about its distribution


## Standard 4-step procedure

- Identify the sample space. Decide what you want to include into your model
- Describe the probability law on that sample space. Ask someone or make necessary assumptions
- Identify the event of interest
- Calculate


## Experiments, outcomes and sample space

- "Randomness" is a way of expressing what we don't know or what is too complicated to calculate precisely
- There are two common interpretations of the word "probability": one is in terms of relative frequency, the second as a quantification of our degree of subjective personal belief that something will happen
- The probability theory is applicable regardless of the interpretation of probability that we use
- The goal is to be able to build probabilistic models
- A **random experiment** is a process by which we observe something uncertain
- An **outcome** is a result of a random experiment
- **Sample space**: set of all possible outcomes of an experiment
- Example: flipping a coin, sample space is `{H, T}`, rolling a die, sample space is `{1, 2, 3, 4, 5, 6}`
- Must be **collectively exhaustive**: no matter how the experiment goes, you are going to receive one of the outcomes from your set
- Must be **mutually exclusive**: if you get head, you don't get tails
- So at the end of the experiment you should be able to point to one and only one element of that set
- When you model, you need to decide what to include and what to leave out. Sample space `(H, T, and it's raining, T, and it's not raining)` is a valid one, if you believe that there is a relation between coin flip and the weather, but you would normally not build such a model
- In case of sequential dice roll (within a single experiment), it would help to build a tree, every leaf correspond to a single outcome
- Once we agreed on a sample space, we need to assign a probability to each of the outcomes
- To do that, someone needs to tell you the rule
- For example, in case of a fair coin, the rule is "H and T are equally likely", but this is not given!
- Instead of individual outcomes, we might be interested in subsets of the outcomes: **events**
- We say that event `A` occurred if the outcome belongs to the given subset
- For example, when rolling a die twice, the event could be "sum of the dice is an odd number", which would include outcomes like `(1, 2)`, `(4, 3)` etc.
- Note: once you identified the sample space and assigned probability to every individual outcome, it is really easy to find probabilities of events
- When all outcomes are equally likely, the calculation is particularly trivial, `P(A)` is simply number of elements in `A` to total number of items


## Continuous sample space and events

- The sample space does not have to be finite or discrete: imagine you are throwing a dart, the coordinates of a point you hit are continuous, and there is an infinite number of points you can hit
- The probability of hitting a single point is zero
- In this case, you don't assign probability to individual outcomes, but directly to events
- Paradoxically, even though probability of any outcome is zero, any time you throw the dart, you will get some outcome: "expect the unexpected"
- You can have discrete but still infinite sample space, which might require special rules
- For example, if your experiment is to "throw the coin until you get H, then you throw it one more time". You might require throwing the coin a million times before you get H, you don't know how many times you need to do it


## Probability axioms

- (1) Probabilities should be non-negative `P(x) >= 0`
- (2) Probability of an entire sample space is 1
- (3) If `A` and `B` have no common elements (disjoint, i.e. do not intersect), meaning `P(A&B) = 0`, then `P(A or B) = P(A) + P(B)`
- As a result, probability is a number between 0 and 1
- `P(A*) = 1 - P(A)`, where `A*` is a complement of `A`
- If `A` in `B`, then `P(A) ≤ P(B)`
- In general, `P(A or B) = P(A) + P(B) - P(A&B)`
- Union bound: `P(A or B) ≤ P(A) + P(B)`


## Conditional probabilities

- When we are given new information, we should revise our beliefs
- If we are calculating `P(A)` and `A` and `B` intersect, and someone tells us that `B` has happened, we are going to switch to calculating `P(A|B)`: probability of `A` given that `B` occurred
- Imagine `A` and `B` as two areas that have some common area `(A&B)`
- Essentially, your sample space gets reduced to `B`, and we are just looking at probability of landing inside an intersection area `(A&B)` within `B`
- Using that image, it can be easily seen that `P(A|B) = P(A&B)/P(B)`, when `P(B) > 0`
- When every outcome is equally likely, this is simply the area of intersection to the area of `B`
- Re-arranging the expression, we get the **chain rule** for conditional probability: `P(A&B) = P(B)*P(A|B) = P(A)*P(B|A)`
- Naturally, `P(B|B) = 1`
- So, conditional probabilities are just like normal probabilities, they simply "happen in a new universe" where `B` is already occurred
- In a way, a condition is a statement about the world, not a special property of a probability
- Just like before, the condition can be an event, not just a single outcome, e.g. "out of 2 dice rolls, the smallest result is 2", which includes many possible outcomes
- Another useful way to visualize this is a tree with branches corresponding to conditions
- Using the tree representation, every subtree is a new universe, with probabilities adding up to 1
- The **law of total probability**: if we split the whole sample space into a disjoint set `A1...An`, covering the entire sample space, then `P(B) = sum of P(Ai)*P(B|Ai) for all i in [1, n]`


## Inference (Bayes rule)

- Using formulas above, you can "reverse conditions" and calculate `P(A|B)` from `P(B|A)`: `P(A|B) = P(A&B)/P(B) = P(A)*P(B|A)/P(B)` (Bayes rule)
- Formulas are just formulas, but what does this mean for our modeling of a real world? Remember, we are bringing stuff into our model for a reason
- _My thought: this is a classical real world → math space → back to real world trip, expecting to land :D_
- In Bayesian interpretation, probability expresses a degree of belief in an event
- With that idea, we might be looking at the conditional probabilities because we believe there is a causality between `A` and `B` (`A` causes `B`)
- In that case, the model we are trying to build would have as a goal to infer `A` from `B`
- `P(B|A)` is a **Prior probability**
- `P(A|B)` is a **Posterior probability**
- One example is a plane and the radar, but the same can be applied to email text and spam, or a disease and test results
- `A` is "there is a plane in the sky", `B` is "radar shows signal", `P(B|A)` is "there is a signal on the radar when plane is flying", `P(A|B)` is "there is actual plane in the sky when radar shows signal", and we are trying to predict whether there is a plane in the sky by looking at the radar
- The tree will have branches `A` and `!A`, branch `A` will have 2 branches `B|A` (true positive) and `!B|A` (false negative), branch `!A` will have 2 branches `B|!A` (false positive) and `!B|!A` (true negative)
- Despite conditional probabilities are just normal probabilities, interpretation can be quite tricky and counterintuitive
- You can notice that the low false alarm ratio can still produce a quite bad device in case of extremely rare event (same when diagnosing an extremely rare disease)
- I.e. `P(B|!A)` is low, but `!A` is extremely high (usually there is no plane in the sky)
- Explanation: `P(A|B) = P(A)*P(B|A)/P(B)`, but `P(B) = P(A)*P(B|A)+P(!A)*P(B|!A)`, and `P(!A)` will drive the `P(!A)*P(B|!A)` high despite `P(B|!A)` being low


## Independence

- When we toss a coin 3 times, the probability of getting H at any individual toss does not depend on results of previous tosses
- So informally, events are independent if `P(B|A) = P(B)`
- `P(A&B) = P(A)*P(B|A)`, so for independent events, `P(A&B) = P(A)*P(B)`
- Since this a very nice and beautiful property, `P(A&B) = P(A)*P(B)` is more useful as a definition of independent events
- One of niceties is: this definition works even when `P(A)=0`
- In a way, independence is a special case of conditional probability
- There are two ways this can be helpful
- On one side, if you believe 2 experiments are dependent, knowing the result of the first one gives you some knowledge about the second one
- In the example with planes and the radar, if you want to predict what the radar shows, it is very useful for you to look to the sky first, and to check whether you can see a plane
- On the other side, if you are trying to figure out whether there is a relation of causality between two events, the formulas give you the way to test it
- For example, you might believe the birds flying low indicates rain, and use these formulas to test your hypothesis
- For that, measure `P(A)`, `P(B)`, then `P(A&B)` and see if the equation holds

### Conditional independence

- Conditional independence: `P(A&B|C) = P(A|C)*P(B|C)`
- Again, conditional independence is not special, it is simply the normal independence in the universe where `C` occurred
- It can happen that `A` and `B` are independent events, but conditionally dependent
- It can happen that `A` and `B` are dependent events, but conditionally independent
- There are many situations, and it is not super intuitive
- One example is `A`="dog barks", `B`="cat in a bad mood", `C`="cat hides". `A` and `B` both cause `C`. However, if you know that `C` has happened (cat hides), then `A` and `B` are no longer independent. Explanation: given the cat that hides, hearing the dog bark will make you believe this has nothing to do with the cat's mood

### Conditional dependence

- Imagine you have 2 biased coins, one with probability of heads being 0.9, another 0.1
- Imagine you randomly pick one with probability 1/2, and then throw it 11 times
- Before you start tossing, the probability of 11th toss being head is 1/2 * 0.9 + 1/2 * 0.1 = 1/2 (since you don't know which coin you are going to pick)
- But if you toss the coin 10 times, and got 10 heads, this gives you a very important inside
- Although with both coins you may happen to see 10 heads in the row, it's far more likely to get this result with the first coin
- This is another case of inference problem: given 10 heads in a row, what is probability of having picked the first coin?
- In such case we will conclude that the probability of tossing heads is ~0.9 (we are pretty sure it's 0.9)

### Pairwise independence vs independence

- Two independent tosses of a coin, the sample space: `(HH, HT, TH, TT)`
- Event `A`: "first toss is H" `(HH, HT)`
- Event `B`: "second toss is H": `(HH, TH)`
- `P(A) = P(B) = 1/2`
- Event `A&B`: "first and second tosses are both H": `(HH)`
- `P(A&B) = 1/4`; `P(A)*P(B) = 1/2 * 1/2`; so `P(A&B) == P(A)*P(B)`, proof that the events are independent
- Event `C`: first and second toss give the same result `(HH, TT)`
- `P(C)` = 1/2
- `P(C&A)` = 1/4
- `P(A&B&C)` = 1/4
- `P(C|A&B)` = 1
- `P(C) != P(C|A&B)`, so `A`, `B` and `C` are not independent, even though pairwise they are!
- We already saw that `A` and `B` are independent, now
- `P(A&C) = 1/4`; `P(A)*P(C) = 1/2 * 1/2 = 1/4`; `P(A&C) = P(A)*P(C)`
- So `A` and `C` are also independent, the same for `B&C`


## Counting

- When all outcomes are equally likely, `P(A)` is the number of elements in `A` to total number of items
- In such a case, the task is really easy if you can count total number of items
- Counting the total number of different outcomes can be difficult, and leads us to the domain of combinatorics
- **Permutation** is an arrangement of set members into a sequence (order matters)
- Number of permutations of n-element set: `n!`
- Explanation: we have `n` choices for the first element, then `n-1` choice for the second, all the way down until we only have 1 choice for the last element
- Number of possible subsets of n-element set: `2^n`
- Explanation: for each element, we decide either to include or not
- **k-combination** of a set `S` is a subset of `k` distinct elements of `S` (order does not matter)
- Number of combinations of `k` elements from an n-element set: `n!/(n-k)!k!` ("n choose k")
- Example: taking random 5 cards from 52 card deck, there are `[52 choose 5]` possible hands

### Sampling table

- Choose `k` objects out of `n` object:
- With replacement, order matters: `n^k`
- Without replacement, order matters: `n*(n-1)* ... *(n-k+1)`
- With replacement, order does not matter: `[n+k-1 choose k]`
- Without replacement, order does not matter: `[n chose k]`

### Counting example 1

- Task: find the probability that throwing fair dice 6 times gives 6 different numbers
- Since the dice is fair, this probability can be calculated as number of outcomes that satisfy the condition to the total number of outcomes
- Total number of outcomes: `6^6` (every time you have a choice of 6 numbers, repeat 6 times)
- All outcomes that make the event are basically permutations of set `{1,2,3,4,5,6}`: you can use these numbers once in any order
- There is `6!` permutations
- So the answer is `6!/6^6`
- Note: it might be confusing to think about permutations here, as you might think "order does not matter, so all my valid outcomes are the same combination", which is true

### Counting example 2

- Task: find the probability that throwing a coin 6 times gives exactly 2 heads, given `P(H) = p`
- For any outcome that satisfies the result, the probability is the same
- `P(HHTTTT) == P(TTHHTT) == p^2*(1-p)^4`
- So the question is: how many of these outcomes can we receive?
- Think of set of positions that we could "insert" H: `{1,2,3,4,5,6}`
- By definition of a task, we are going to pick only 2 of these positions, picking more positions means we are going to receive more than 2 heads
- Number of ways you can pick 2 positions out of 6 is the number of combinations `[6 choose 2]` and = `6!/(6-2)!2!`
- The answer is `p^2*(1-p)^4 * 6!/(6-2)!2!`

### Counting example 3

- Task: throwing a coin 10 times, we are told that there was exactly 3 heads (event `B`). What is the probability that first 2 tosses were heads (event `A`)?
- We don't know `P(H)`, and given the results, the coin is probably not fair
- Since the coin is not fair, not every outcome is equally probable
- BUT: we were told that there were exactly 3 heads, which brings us to the conditional probabilities
- We transfer to a new universe, where B has already happened. And in that universe, all the outcomes are equally likely
- Explanation: whatever `P(H)` is, `P(HHHTTTTTTT)` is the same as `P(TTTTTTTHHH)`
- Since the outcomes are equally likely, we just need to divide number of outcomes where 2 out of 3 heads are first tosses to the number of outcomes in `B`
- Number of outcomes in `B` can be defined the same way as in example 2, by picking positions, it is `[10 choose 3]`
- Outcomes where 2 first tosses are heads differ only by the position of the third head, there are 8 possibilities
- The answer is `8/[10 choose 3]`


## Random variables

- Random variable is an assignment of a numerical value to every possible outcome
- In other words, a random variable is a numerical description of the outcome of a statistical experiment
- Hint: it's a `map`
- Example: pick student at random, map to the student's height
- Essentially it is a function from the sample space to real numbers
- You can have more than one random variable for the same sample space, e.g. `H` for height, `W` for weight
- Functions of random variables are also random variables
- For example, height in inches is a function of height in cm, which is a function of a student
- Random variables can be discrete or continuous (height is the example of a continuous random variable)


## Discrete random variables

- For every value of a random variable, we could say how likely it is to obtain that value
- Example: user experience score that can take values "Satisfied", "Tolerated", and "Frustrated" with probabilities 1/2, 3/10 and 1/5 (respectively)
- `px` is a function that assigns probability of random variable `X` to take value of `x`
- Formally, `px(x) = P(X = x)`
- `px` is a **probability mass function (PMF)**, also known as the discrete density function
- Sum of all probabilities for each `x` is, of course, 1 (since every outcome will map to a certain x by the random variable)

### Binomial distribution

- **Binomial distribution** with parameters is the discrete probability distribution of the number of successes in a sequence of n independent experiments (aka Bernoulli trials), each asking a yes–no question, and each with its own boolean-valued outcome: success (with probability `p`) or failure (with probability `q = 1 − p`)
- **Bernoulli distribution** is a special case of the binomial distribution where a single trial is conducted. It is the discrete probability distribution of possible outcomes of a single experiment that asks a yes–no question, with probabilities `p` and `q`
- **Negative binomial distribution** is the discrete probability distribution of the number of failures in a sequence of independent and identically distributed Bernoulli trials before `r` successes occurs


## Expected value of a random variable

- If you interpret probabilities as frequencies, you can calculate an "average" value attained by the random variable
- Example: you play the game where you win `$1` every 6 times you spin, `$2` every second time you spin and `$4` every third time you spin. On average, every spin earns you `$(1/6*1 + 1/2*2 + 1/3*4)`
- This is the **expected value** of a random variable, denoted as `E[X]`
- `E[X] = Sum(x*P(X = x)) over all x`
- One interpretation: the average you get over a large number of experiments
- This is exactly how you would calculate an average of series of regular numbers using the values weighted by actual frequencies: `avg(1,1,1,1,1,3,3,5) = 5/8*1 + 2/8*3 + 1/8*5`
- Another, "physical", interpretation: the "center of gravity" of the PMF diagram
- One of the most important properties of expectation is **Linearity of expectation**: `E[X+Y] = E[X] + E[Y]` (true even if `X` and `Y` are dependent)
- If you have a random variable `X` and a random variable `Y = g(X)`, you can calculate `E[Y]` as `Sum(g(x)*Px(x)) over all x`
- This is called **the law of the unconscious statistician (LOTUS)**
- Harder way to calculate it: `Sum(y*Py(y)) over all y`
- (!) Caution: in general, average of a function does not equal function of an average, `E[g(X)] != g(E[X])`; that is to say these 2 operations do not commute
- In other words, in general, you "cannot reason on the average"
- Some exceptions when you can reason on the average: `E[c] = c`, `E[c*X] = c*E[X]`, `E[c*X + b] = c*E[X] + b` (basically, linear functions)


## Variance of a random variable

- **Variance** `Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2`
- Variance is a measure of how "spread" the values are
- One annoyance working with variance is having unit squared
- To solve it, you could switch to **standard deviation** (square root of variance)
- Some special cases for variance: `Var(aX) = a^2*Var(X)`, `Var(X + b) = Var(X)`


## Conditional PMF

- Just like with normal probabilities, conditional PMF brings us to the new universe where the event has happened
- All you need to do it to throw away part of the old universe that is not relevant anymore and adjust the probabilities of the remaining values, so that they add up to 1, keeping their relative proportions
- Expected value, variance etc. is then calculates same as before
- `E[X|A] = Sum(x*P(X = x|A)) over all x`

### Conditional expectation and coin tosses

- `X` is the number of tails before the first heads
- `E[X|first toss is tails] = 1 + E[X]`
- Basically, this says, "yes, tossing a coin has no memory, so every time you toss the coin, the probability of getting heads or tails is still the same, 1/2"
- But thanks to having tossed one tail already, `X` is given to be at least 1
- This does NOT suggest that we should "expect heads more and more every time we toss tails"


## Joint PMFs

- Imagine `X` is the height of a student, `Y` is a weight of a student
- Both tell me something about the class
- But I might be interested to know about their relation, how the height relates to weight
- In this case I might decide to look at the **joint probability** `pxy(x,y) = P(X = x and Y = y) == P(X = x, Y = y)`
- Conditional probability: `P(X = x|Y = y)`
- Conditional probability can be used to calculate joint probability: `P(X = x, Y = y) = P(X = x|Y = y)*P(Y = y)`
- **Marginal probability** is the probability of an event irrespective of the outcome of another variable: `P(X = x)`
- Independence: marginal probability equals conditional probability
- For independent random variables, `E[XY] = E[X]*E[Y]`, another exception calculating expectation
- If `X` and `Y` are independent, `g(X)` and `h(Y)` are also independent
- So, for independent random variables, `E[g(X)h(Y)] = E[g(X)] * E[h(Y)]`
- If `X` and `Y` are independent, `Var(X + Y) = Var(X) + Var(Y)`


## Continuous Random Variables

- Numerical value of a continuous random variable can be any real number
- Instead of PMF we use PDF: probability density function, usually denoted `f(x)`
- Probability of a continuous random variable `X` to fall into interval `a, b` will be the area under the probability density function between `a` and `b`
- So instead of sum we go to the integral
- Since probability cannot be negative, PDF has to be non-negative: `f(x) >= 0`
- Since the total probability is equal to 1, the integral of a PDF over all `x` has to be equal 1 (the area under the curve must be equal to 1)
- Important to note that PDF can take values > 1, as long as the area does not exceed 1
- In other words, the value of PDF is not probability! It's the area that is probability. This is a source of possible confusion
- Any individual point has a zero probability, so the density function doesn't tell you the probability of a specific value, but of small intervals
- The unit is probability per interval
- So when dealing with continuous random variable, forget about individual values, always think in intervals
- Expectation and variance have the same meaning, and the only difference, again, is that we integrate (over PDF) instead of summing

### CDF

- Is there way to treat discrete and continuous random variables in a similar way? We cannot just convert PMF into PDF, since we need to have intervals, and discrete variables are all specific numbers, and the specific numbers in case of continuous random variable all have zero probability
- The way to do it is to use CDF: **cumulative distribution function** (in case of discrete random variable will have distinct steps), usually denoted `F(x)`
- Unlike PDF, CDF expresses probability, specifically the probability that the random variable `X` will take any value `≤ x`, i.e. `F(x) = P(X ≤ x)`
- You can even "cook" a random variable that is neither purely discrete nor continuous, but a kind of combination of two, and CDF will still work well
- In case of continuous random variable, you can get the PDF from CDF by taking a derivative
- The value of the derivative will be considered undefined in the points of "sharp corners"
- Using CDF, we can calculate probability of a random variable to lay in the interval `(a;b]`, using `P(a < X ≤ b) = F(b) - F(a)`
- For a non-negative random variable, discrete or continuous, you can calculate `E[X]` as integral of CDF over all `x`

### Distributions

- **Uniform distribution**: `П`-shape, every interval between `a` and `b` has the same probability `1/(b-a)`, all the rest zero
- Computer random number generator has uniform distribution, and using uniform distribution you can sample from any distribution, using **inverse transform method** (for simulations)
- Basically, `U ~ Uniform(0,1)`, `F` is CDF of any distribution you want, and `G` is inverse of `F`, then `X = G(U) ~ F`
- **Gaussian (or normal) distribution**: the famous [bell curve](https://en.wikipedia.org/wiki/Normal_distribution)
- The standard normal distribution, also called the z-distribution, is a special normal distribution where the `E[X] = 0` and `Var(X) = 1`
- General normal distribution `N(mu, sigma^2)` has `E[X] = mu`, and `Var(X) = sigma^2` and there is a (quite complicated) formula that describes this curve (see Wikipedia)
- If `X` is normally distributed, `Y = aX + b` (linear transformation) is also normally distributed
- For that new curve, `E[Y] = a*mu + b` (see exceptions for reasoning on expectation), and `Var(Y) = a^2 * sigma^2`
- You can *standardize* any general normal distribution `N(mu, sigma^2)` by subtracting `mu` and dividing by `sigma`: `Z = (x - mu)/sigma`
- Subtracting `mu`, the bell curve gets shifted along the horizontal scale to obtain `E[X] = 0`
- Dividing by `sigma`, the variance gets divided by `sigma^2`, obtaining `Var(X) = 1`
- And since this transformation is linear, it also produces a normal distribution
- Less hand-wavy explanation:
- `Z = (X - mu)/sigma = X/sigma - mu/sigma`, `a = 1/sigma`, `b = -mu/sigma`
- `E[Z] = a*mu + b = mu/sigma - mu/sigma = 0`
- `Var(Z) = a^2 * sigma^2 = (1/sigma)^2 * sigma^2 = 1/sigma^2 * sigma^2 = sigma^2/sigma^2 = 1`
- The standardized normal distribution is also called "z-score"
- **Z score** is very useful when processing some experiment results (see machine learning), it is usually recommended standardizing the general normal distribution
- Z score can tell you, for any datapoint, how many standard deviations you are from the mean
- Calculating CDF of a given normally distributed continuous random variable would require calculating the area under the bell shaped curve from minus infinity to a certain value `x'`, that is, integrating from minus infinity to `x'`
- Apparently, there is no closed-form expression for that, so people use tables with pre-calculated values for given `x'`
- The table is pre-calculated for the standard normal distribution, and that is another reason to standardize the normal distribution


## Multiple Continuous Random Variables

- Here we just apply all the previous concepts to the continuous random variables
- It is really all the same things as before, just more difficult to wrap your mind around, because of multiple dimensions and integrals
- For any practical purposes, you also should be able to calculate integrals
- In case of jointly continuous random variables you have 2 (or more) continuous random variables `X` and `Y`, and your PDF becomes a surface on top of 2-dimensional plane
- Instead of small intervals, we work with small areas and the probability is the volume under PDF
- The integral becomes double integral
- You can go from joint PDF `f(x,y)` to marginal PDF `f(x)`, by removing integral by `x`, and integrating only by `y`
- Continuous random variables are independent if their joint PDF factors out as a product of their marginal PDFs: `f(x,y) = f(x)*f(y)`
- The concept of conditionality is also the same, except you cannot condition on a given value of `Y` (which has probability of zero), so yet again, you need to work with small intervals
- You write it as `P(x <= X <= x + sigma | Y ~ y)`, the probability of `X` to fall into a small interval `[x, x + sigma]` given `Y` is "infinitesimally" close to `y`
- Conditional PDF: `f(x|y) = f(x,y)/f(y)`


### Monte-Carlo method

- Since probabilities are calculated using integrals, we could go the opposite way and calculate (or, rather estimate) integrals using probabilities
- Trivial explanation: instead of calculating probability of hitting the circle with a needle using a formula that involves Pi, we could estimate Pi by throwing millions of needles and see how many land inside the circle


## Continuous Bayes rule

- The idea is the same as with normal discrete probabilities
- The main goal is to find out the distribution of unknown quantity given distributions of already observed random variables
- Given: `p(x)` and `f(x)`; `p(y|x)` and `f(y|x)`, in various combinations
- To find: `p(x|y)` and `f(x|y)`
- Example: you measure the current in the circuit, using some analog device. This device can pick up on some noise. When you look at the device, the value depends on the current, but also on the noise. You are trying to guess what the actual value of the current is
- _My thought: this is the classical problem in neuroscience_
- In case of continuous random variable, the formula looks the same: `f(x|y) = f(x)f(y|x)/f(y)`
- The discrete and continuous probabilities can mix
- Classic digital communication problem: you transmit 0 or 1, and have some gaussian noise added to the signal, resulting in continuous signal. Zeroes and ones have a certain distribution, depending on the message, and final signal has a certain density function, different for (conditional on) 1 and 0 (two small hills, one with center at zero and one with center at one)
- You can calculate probabilities of the signal to be 1 or 0 as follows: `p(x|y) = p(x)f(y|x)/f(y)`
- Of course, in real world, this can be a bit trickier than that
- For example, to know the value of PDF, you might need to calculate an integral, and there are nasty integrals to calculate


## Derived distributions

- Find the distribution of a random variable, that is a combination of other random variables (with known distributions) with functions applied on those variables
- For example: given distributions of 2 random variables, find the distribution of their ratio
- Or, even simpler: given `X`, uniformly distributed on `[0, 2]`, find PDF of `Y = X^3`
- In case of continuous random variables, there is a mechanical process that you need to follow:
- 1). Find the CDF of `Y`
- Example: `F(y) = P(Y <= y) = P(X^3 <= y) = P(X <= y^(1/3)) = 1/2y^(1/3)`, 1/2 because `X` is uniformly distributed on `[0, 2]`, so `P(X <= y^(1/3))` is just an area of a rectangle with length of `y^(1/3)` and height of `1/2`
- 2). Differentiate
- Example: `f(y) = dF(y)dy = 1/6y^(2/3)`
- We know that `Y` can only take values in interval `[0, 8]`, so cut off the result at 0 and 8
- So it is simple if you can differentiate :)


## Covariance and correlation

- **Covariance** measures the direction of the relationship between two variables
- A positive covariance means that both variables tend to be high or low at the same time
- A negative covariance means that when one variable is high, the other tends to be low
- Shortcut formula: `Cov(X, Y) = E[XY]-E[X]E[Y]`
- When 2 variables are independent, then `Cov(X, Y) = 0` (the converse is not necessarily true, i.e. can still be 0 for 2 dependent variables)
- When 2 variables are fully dependent, i.e. `Y` is `X`, covariance is just a variance, so `Cov(X, X) = Var(X)`
- Similar problem as with variance, covariance has a unit that is product of unit of `X` and `Y`
- This is why, in practice, it is more convenient to work with **correlation**
- To calculate the correlation, we standardize both random variables and then take the expectation of a product (of those standardized variables)
- We obtain `Cov(X, Y)/(sigmaX*sigmaY)`, which does not have any units and nicely falls in `[-1, 1]` interval
- Covariance measures the direction of a relationship between two variables, while correlation measures the strength of that relationship
- When 2 variables are independent, then correlation is = 0 (the converse is not necessarily true)
- **In other words, the correlation does not imply causality!**


## Conditional expectation and variance

- We might want to calculate `E[X|Y=y] = g(y)` (expectation of `X` given `Y` takes value of `y`), and that would be a specific number
- But it might be even more interesting to look at `E[X|Y] = g(Y)`, expectation of `X` for all possible values of `Y`, this would be a function of a random variable `Y`
- As we have seen, functions of random variables are also random variables
- So the expectation of a random variable, conditional on another random variable, `E[X|Y]`, is also a random variable
- And since we are having so much fun, why don't we keep building on top?
- Voilà: `E[E[X|Y]] = E[g(Y)]`
- If we try to use the formula for the expectation, we'll discover the **law of iterated expectations**: `E[E[X|Y]] = E[X]`
- _My thought: this is another real world → math space → back to real world trip_
- Same logic applied, let's move from `Var(X|Y=y)` to `Var(X|Y)`, the conditional variance will also be a random variable
- **Law of total variance**: `Var(X) = E[Var(X|Y)] + Var(E[X|Y])`
- The proof is "one of those proofs that do not convey any intuition". Yet another math trip, where formulas just do their magic
- Note: this formula can also be used (in certain circumstances) as a shortcut to avoid doing integrals

### The story behind the formula

- Imagine 100 different drivers somehow randomly selected to race on 1 of 3 different tracks, and we record their best lap times, obtaining 100 values
- We repeat that experiment every day for 10 years, and record `365*100*10` values
- The time of the best lap of a driver at a given track is a random variable `X` with certain variance, and depends on the skill of a driver
- The track, since it is randomly selected, is another random variable `Y`
- Imagine the tracks are quite different: some more technical and long, some fast and short, so the average lap time varies greatly and depends on the type of the track
- Intuitively, the total, unconditional, variance, for those `365*100*10` values, depends both on the driver skills and the type of the track
- Getting back to the formula:
- `E[X|Y]` is an average time on a track, for each of 3 tracks
- It is basically the function "`if Y = 1 then x1; else if Y = 2 then x2; else x3`"
- `Var(E[X|Y])` expresses how different the tracks are (how different the track averages are, for each of 3 tracks), this is a number
- `Var(X|Y)` expresses how different the drivers skills are, on each of 3 tracks, this again is a function
- `E[Var(X|Y)]` is an average difference in drivers skills, across all tracks, this again is a number
- So the formula expresses the same what we argued intuitively: the final difference in the results is a sum of differences between tracks and drivers skills
- This is the fundamental concept for applying the probability theory to the real world: just replace the tracks with measurements, and the difference in drivers skills with an error


## Moment Generating Functions (MGF)

- MGF is an alternative way to describe a distribution, in addition to PDF and CDF
- `E[X^n]` is a `n`-th moment
- MGF of a random variable `X` is a function `M(t) = E[e^tX]`
- Why? Apparently, if you expand `E[e^tX]` as Taylor series, you can see that the series contain all the moments from 0 to infinity
- And the way Taylor series work, you can use MGF to "generate" `n`-th moment `E[X^n]` by taking an `n`-th derivative of `M(t)` and evaluating it at zero
- This property makes MGFs useful in several ways
- MGF "uniquely determines the distribution", i.e. if two random variables have the same MGF, then they must have the same distribution
- MGF also simplifies working with sum of random variable
- If `Mx` is MGF of `X`, and `My` is MGF of `Y`, and `X` and `Y` are independent, then MGF of `X+Y` is just a product `Mx*My`
- Not all random variables have MGFs


## Weak law of large numbers

- Setup: there is about 20M penguins in Antarctica, so the total size of the **population** is 20M
- If we measure the height of each and every one, we will get a series of numbers with mean `mu` and standard deviation `sigma`
- Unfortunately, measuring all 20M penguins is too hard
- Instead, we will go on an expedition, catch some small **sample** of `n` penguins (randomly) and measure their height
- This is how the precise task with definite answer becomes a domain of the probabilistic study
- Catching one penguin randomly is an **experiment**
- The result can be any penguin out of 20 million, so the **sample space** is 20 million possible penguins
- Mapping a penguin to its height is a **random variable**
- So when someone says "draw a random variable", think of catching a penguin randomly
- Thus, our expedition will produce a sequence of random variables `X1, X2, ... Xn`
- All these random variables will be **identically distributed**, as they come from the same population
- We assume these random variables are also **independent**, meaning we catch every penguin independently (in practice, needs to be proven)
- Technically speaking, we also need to make sure that, once we measure a penguin, we release it back into the environment so that the same penguin can be caught again (we draw randomly **with replacement**)
- If we average the height of the penguins in the sample, we will obtain the **sample mean** `Mn = (X1 + X2 + ... + Xn)/n`
- The sample mean is a function of random variables, so it is also a random variable
- We can also calculate `E[Mn]`, expectation of the sample mean
- The expectation of the sample mean is a number
- The sample mean is what we get from 1 expedition, the expectation of the sample mean is an average value of the sample mean over a (very) long series of expeditions
- It can be shown that `E[Mn] = mu`, the expectation of the sample mean is the true mean
- `Var(Mn) = sigma^2/n`, with variance getting the smaller and smaller with larger `n` (having the large sample size, in a way, removes randomness from your experiment)
- Now, this is all cool, but we still don't know `mu`, and we don't know how to calculate `E[Mn]`
- **Weak law of large numbers:** `Mn` (sample mean) converges in probability to `mu` (true mean)
- This can be proven using Chebyshev inequality (see below)
- In plain English, we can interpret it in the following way: the sample mean is a "good estimation" for the true mean (the population mean)
- We have to agree what "good" means
- Usually we want to claim that the sample mean is less than epsilon away from the true mean with probability `P(|Mn-mu|≤epsilon)` or more
- `|Mn-mu|≤epsilon` is an **accuracy**
- Desired probability is **confidence**
- You can use Chebyshev inequality to calculate the necessary `n`, i.e. the minimal size of the sample required to make that claim
- This is usually used in polls, for example, to find out what fraction of population prefers Coke to Pepsi
- Usual numbers people use in such polls is 3% of error and 95% of confidence
- Using the law, we might establish, for example, that in order to conduct a poll with these guarantees, we would need 10K participants

### Chebyshev inequality (for reference)

- **Markov inequality**: `E[X] >= a*P(X>=a)`, follows from the formula of `E[X]` using math magic
- Similar for the variance, **Chebyshev inequality**: `Var(X) >= a^2*P(|X-E[X]|>=a)`
- Rewriting the last statement: `P(|X-mu| >= k*sigma) <= 1/k^2`, where `mu = E[X]` and `sigma` is standard deviation, and `k` is positive number
- This formula expresses the probability of being `k` standard deviations away from mean

### Convergence in probability

- Sequence `an` converges to a number `a`, if "`an` eventually gets and stays (arbitrarily) close to `a`", i.e. `lim(an) = a`, with `n->infinity`
- Another way to say it: for every `epsilon > 0`, there exist `n0` such that for every `n >= n0` we have `|an-a| <= epsilon`
- So regardless of how small the `epsilon` is, all we need to do is to go sufficiently far in the sequence, and there will be the point after which we won't get any number larger than `epsilon`
- Sequence `Yn` of random variables converges **in probability** to a number `a`, if (almost all) of the PMF/PDF of `Yn` eventually gets concentrated (arbitrarily) close to `a`
- I.e. the probability that a random variable falls outside the certain band a converges to zero
- This is very abstract, but here is a simple example:
- We flip the fair coin, and record 1 or 0 for heads or tails (random variable `X`)
- We do that many times, obtaining `X1, X2, ... Xi ... Xn`
- Every time we use all the previous results to calculate the average `Mi = (X1 + X2 + Xi)/i`
- This average is a random variable, as it is a function of random variables
- As a random variable, `Mi` is going to have a certain distribution, we are not expecting it to be strictly 0.5
- In the beginning, this distribution will be quite spread between 0 and 1, but if we produced the new distribution for every new coin flip, every new distribution would be narrower and narrower
- With infinite number of coin flips, the distribution will become infinitely narrow


## Central Limit Theorem

- Given i.i.d `X1, X2, ... Xn`, with finite mean and variance
- `Sn = X1 + X2 + ... + Xn`
- `Zn` is standardized `Sn`, so `E[Zn] = 0`, `Var(Zn) = 1`
- By definition, `Zn = (Sn - E[Sn])/sigma(Sn)`
- Using expectation and `sigma` of sample mean, as discussed in "weak law of large numbers", `Zn = (Sn - n*mu)/sqrt(n)*sigma`
- `Z` is standard normal random variable
- The theorem: for every `c`, `P(Zn <= c) -> P(Z <= c)`
- `P(Z <= c)` is a standard normal CDF, well known and pre-calculated in form of tables
- `P(Zn <= c)` is a CDF of our standardized cumulative random variable
- So with `n` "big enough", you can replace CDF of `Zn` with simply CDF of `Z`
- Note: this is only true for CDFs, not PDFs or PMFs! FYI, there are versions of the theorem, that, given some extra assumptions about `Xs`, produce similar statements for PDFs or PMFs
- The key point is that it does not matter what the distribution of `Xs` is, and it does not matter whether `Xs` are discrete, continuous, or mixed, as long as it is identical, and we know its mean and variance
- This provides a nice computational shortcut: when we encounter some expression that has a form of `Zn`, we can immediately calculate it using `Z`
- In practice, this allows us to pretend that `Sn` is normal, even if the theorem does not actually say that, strictly speaking
- For example, this is used to make assumptions about noise
- Whenever you have a phenomenon which is noisy, and the noise that you observe is created by adding lots of little pieces of randomness that are independent of each other, the overall effect that you are going to observe can be described by a normal random variable
- Note: this statement includes independence assumption, but the variables also need to be "identically distributed"
- Example: Brownian motion, modeling the displacement of a particle as a result of collisions with liquid molecules
- Each time the molecule of liquid hits the particle, the velocity is the random variable, the collisions are independent and, since they all come from the same liquid volume, identically distributed
- The displacement will be a random variable, and by the theorem, it will be normally distributed
- Some people use the same assumption to model the financial market movements (which led to a spectacular failure)

### Application of CLT in poll size calculation

- Another way this can be used is to calculate the required sample size for a poll
- Suppose we want `P(|Mn-mu|≤epsilon) > 0.95`, i.e. `P(|Mn-mu|>=epsilon) ≤ 0.05`
- Event of interest, `|Mn-mu|`, is `(X1 + X2 + ... Xn - n*mu)/n`, so we want `(X1 + X2 + ... Xn - n*mu)/n >= epsilon`
- Or, `(X1 + X2 + ... Xn - n*mu)/(sqrt(n)*sigma) >= epsilon*sqrt(n)/sigma`
- The left part has now a shape of `Zn`, and can be replaced by `Z`
- Now we are calculating the probability `P(|Z|>=epsilon*sqrt(n)/sigma)`
- There are only 2 unknowns left: the sample size `n` and `sigma`
- However, we know that sigma is at max 0.5, given the poll can only produce values of 0 or 1, so we can use a "conservative" value of `sigma = 0.5`; using the upper bound we know that `P(|Mn-mu|>=epsilon)` is, at max, `P(|Z|>=0.2*sqrt(n))`
- Since `Z` is symmetrical, `P(|Z|>=xxx) = 2*P(Z>=xxx)`, and `P(Z>=xxx) = 1 - P(Z<=xxx)`, which is CDF for `Z`, and can be found from tables
- `P(|Z|>=0.2*sqrt(n)) = 2(1-P(Z<=0.2*sqrt(n)))`
- `2(1-P(Z<=0.2*sqrt(n)))` is the max value for probability of error exceeding epsilon for sample size `n`
- Given our target value for `P(|Mn-mu|>=epsilon)`, we can also find the actual required `n`
- Using the tables, we can find out what value we need `0.2*sqrt(n)` to be to produce desired value of `P(Z<=0.2*sqrt(n))`
