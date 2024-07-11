# AI Explainability
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Stanford: Machine Learning Explainability Workshop](https://www.youtube.com/playlist?list=PLoROMvodv4rPh6wa6PGcHH6vMG9sEIPxL)

## Intro

- Not all AI applications require interpretation. For example, suggesting similar products on Amazon: as long as it works, and generates revenue, you might not really care how it does what it does
- And many such algorithms are already proven and tested, so you can more or less take it off the shelf and just trust it
- But ML is increasingly being employed in settings where stakes are high (health, justice, finance etc.)
- At the same time, the models that are being deployed in those settings may not be that well tested/validated and understood, which is a risky combination
- In that scenario, accuracy alone is not enough; and the new legislation is raising to ensure nondiscrimination and safety
- For an end user affected by the algorithm decisions, the right to explanation is very important
- How do you even quantify things like fairness? There is an inherent incompleteness of the problem formalization
- Explainability can be useful even for debugging: the model may make a right prediction (e.g. "this is a cat"), but use the wrong features to do it (e.g. "the milk"), which would be helpful to correct the model
- But it would be especially important in life-changing scenarios (e.g. "approve loan", "likely criminal offender"), in that case the model prediction is not enough, you need to know how that prediction was made (e.g. "why loan was denied")
- First approach to achieve interpretability is to build **inherently interpretable** models (such as decision trees, provided there are few branches); this is good if your problem fits into such a model
- Second approach is to explain models in a **post-hoc** manner
- There exist open-source libraries that provide interpretable models and explanation methods; many tech startups focus specifically on this topic


## Inherently interpretable models

- Here we'll consider a bunch of models developed around 2015-2016
- **Bayesian rule list** is a long list of "if-else" rules, produced by the model
- Unlike decision trees, this model does not use greedy approach but instead relies on probabilities
- As usual, you assume a particular process that generates the data, the parameters are unknown and you are trying to estimate them from data
- In practice it means, you sample the length of the list, the rules, the "if" part, the "then" part, i.e. everything
- **Interpretable decision sets** is very similar, with the main difference: instead of a long "if-else" list (a single statement), you get a long list of statements in the shape of `if <p1> and <p2> and ... and <pn> then <decision>`
- While this may sound like a purely stylistic difference, this makes a huge difference for a person who needs to interpret the results (the amount of cognitive load required to analyze the decision list)
- Interpretable decision sets tries to optimize, in addition to recall and precision, for parsimony, distinctness, and class coverage (the last 3 are important for interpretability)
- Parsimony means: minimize the number of rules in a set and the number of predicates in each of the rules
- Distinctness means: avoid duplicate rules (minimize the number of points that satisfy more than one rule)
- Class coverage means: there is a rule for every class
- **Risk scores**: for a given datapoint (e.g. person), through the list of conditions (e.g. "age > 40"); every condition, when matched, adds or removes a certain number of points to the total score, the final score is used to make a decision (e.g. "give loan")
- This is popular in health and criminal justice, with list of conditions being manually crafted by domain experts
- Doctors and judges like it as it is very easy to apply
- The analogous AI model would learn the list of conditions from data
- **Generative additive models (GAMs)**: model output as a function of input variables (e.g. linear model)
- "Additive" means: add several functions of different input variables together
- Interpretability comes from the visualization of the function (but is questionable)
- **Prototype selection**: identify K prototypes from the dataset (data points that represent a given class the best), so that, when given a new datapoint, we could classify it by its similarity with one of the K prototypes (with a high probability)
- Each prototype covers an epsilon-bound neighborhood around it
- This is very similar to nearest neighbors
- **Prototype layers** combine the prototype selection approach with DL by adding an extra layer in front of a fully-connected layer(s)
- The prototype layer computes the distance between a current datapoint and each of the prototypes selected so far
- The fully connected layer is using the distances to make the prediction
- The model is trained end-to-end, learning to classify but also good prototypes
- At a prediction time, this allows to point to the prototype that influenced the decision (_my note: I have some doubts about it. I have no idea what FC layer could have done with my distances. And evaluation of these layers, as discussed later, shows that the prototype layers do not always produce meaningful representations_)
- _My note: this also looks a bit like an auto-encoder, as there is a bottleneck in a form of a low-dimensional prototype layer, and also sounds a bit like a latent variable model. I guess all of these ideas are connected at some deeper level_
- **Attention based models** explicitly models the importance of the parts of the input
- This allows to see what model paid the attention to, when making a decision


## Post-hoc explanation

- In this setup we cannot change the model
- The explanation should be both understandable and faithful (actually telling the truth about how model works)
- What could you provide to the user? Model parameters? Example predictions? Summarize the model parameters using a rule tree? Select the most important features/datapoints that affect the prediction? Describe how to flip the model prediction?
- All of these could be useful, depending on the model and the user
- **Local explanations** explain individual predictions. Help to unearth biases in the local neighborhood of a given datapoint
- **Global explanations** explain the complete behavior of the model. Help to shed light on big picture biases affecting larger subgroups

### Local explanation methods

- **Feature importance** methods: LIME, SHAP
- LIME: take a single datapoint `x`, sample points around `x` (e.g. add gaussian noise), use model to predict the label for each sample, weight samples according to distance to `x`, learn simple linear model on weighted samples, use that linear model to explain
- Kind of gives you the most important part of a boundary that affected the classification
- SHAP: estimate the marginal contribution of each feature to the prediction (how much does the prediction change when feature `x` change?), averaged over all possible permutations of features (`x1`, `x1, x2`, and so on)
- Very computationally expensive
- **Rule-based** methods: Anchor explanation
- Take a single datapoint `x`, sample points around `x` (e.g. add gaussian noise) to generate a local neighborhood, identify an "anchor" rule which has the maximum coverage of the local neighborhood and also achieves high precision
- The result is the big "if-and-and-and-then" statement, that works as an explanation
- **Saliency maps**: input-gradient, SmoothGrad, integrated gradients, Gradient-Input
- Input-gradient measures how much output changes when an input feature change, gives you a heatmap, but it is very noisy and difficult to interpret
- SmoothGrad smoothens the gradient computation, much less noisy
- Integrated gradients method is even better
- **Prototype/example based** methods (synthetic or natural examples): influence functions, activation maximization
- Influence functions: identify datapoints in the training set that are responsible for the prediction of a given test datapoint (can be expensive to compute)
- Activation maximization: identify examples that strongly activate a function of interest
- _My note: synthetic examples are quite cool_
- **Counterfactual** methods: which features need to change and how much in order to change the prediction
- Could be useful to know what you would need to change (and by how much) to get the loan
- Minimum distance counterfactuals are pushing the point to the decision boundary using the shortest path (choice of distance is important)
- Least cost counterfactuals use a notion of cost instead of a distance (it's easier to get bigger salary than change your race/gender)
- People are using variational autoencoders to search for the counterfactuals in the latent space

### Global explanation methods

- In a certain sense, the global explanation can be seen as a summary of local explanations; but it would be impractical to do it manually
- **Collection of local explanations**: generate a local explanation for every datapoint (using one of the approaches above), pick a subset of `k` local explanations to use as a global explanation
- Those `k` local explanations should be representative (summarize the behavior) and diverse (avoid redundancy)
- SP-LIME uses LIME + greedily picks `k` local explanations
- SP-Anchor uses Anchor + greedily picks `k` local explanations
- **Representation based** approaches: gain model understanding by analyzing intermediate representations of a model; to see if the model uses the same concepts as a human (e.g. identifies zebra thanks to stripes, meaning it has some notion of stripes)
- TCAV explicitly tests for those concepts (e.g. stripes). You run your model on your actual datapoints (e.g. images of animals) and also on images representing a concept (e.g. images of various stripes), get the inner representation for each of those images (the output of some inner layer) and run a liner classifier on those. You then find a vector that is orthogonal to the linear decision boundary and pointing towards the concept of stripes (in the space of those intermediate inner representations). Once you have that vector, you compute derivatives to determine the importance of the notion of stripes for any given prediction (HOW?)
- This basically allows to express our human-understandable concept of stripes as a vector in a model space allowing to quantify this notion and make a formal analysis
- **Model distillation** approaches: take the data, take predictions of a model (the one that we want to explain) and build a simpler, interpretable model (e.g. GAM, decision tree) that tries to mimic the predictions of our model
- The user could pick a couple of features and ask the model to explain the classification in relation to those features (e.g. by spitting out the "if-else" rules, with the first branch being made at the features of interest)
- **Summaries of counterfactuals**: check how many features you need to change and by how much in order to flip the prediction; for each of the subgroups
- This summary across different subgroups allows detecting whether one of the subgroups would have to do a disproportional amount of work in order to change the label
- This is, again, useful in situations such as loan approval, to detect racial/gender biases (as a white male, I might be only required to get a slightly higher salary, as a black woman, I might be needed to get off drugs and fix my alcohol addiction)
- _My note: I'm not suggesting anything LOL, just describing the example_


## Evaluating interpretations/explanations

- Functionally-grounded evaluation: quantitative metrics (e.g. number of rules, predicates, prototypes, lower is better)
- Human-grounded evaluation: involves human judgement (e.g. simply ask a human which explanation is better or ask a user to make a prediction for a new datapoint based on the description and compare the accuracy of predictions human vs model)
- Application-grounded evaluation: compare to the domain expert (compare the accuracy of prediction between a domain expert vs a domain expert equipped with a model explanation)
- For example, if model explanation suggests the 5 most important features to make a diagnosis, you can tell that to doctors and see if this information (literally the list of 5 most important features, not the model prediction!) improves the accuracy of their predictions
- For some inherently interpretable models (such as "if-else" rule list), the model itself is, essentially, the description
- You can still compare the 2 models in terms of number of rule, predicates etc.
- And you can do the user studies to see which description is easier to understand (randomly assign test users to one of 2 models, let them answer 12 questions, some of them require using model for predictions, some of them asking for human-level descriptions that they would need to extract from the model)
- Prototype layers and attention based models are trickier, do not always produce meaningful representations and harder to evaluate (TODO: there was no real good discussion on this subject)
- For the post-hoc explanations, you need to start by proving the **faithfulness of explanations**
- If you have ground truth (you know which features your model uses), you are more or less in luck, since you could compare your output with that ground truth. Sounds kind of obvious statement
- Things get more interesting when you don't have a ground truth
- If the explanation is itself a model (as in case with LIME, which produces a linear model), you can compare its predictions with the predictions of the underlying model, in the neighborhood of the given datapoint
- Another way: remove the features that your explanation suggests being important, see what happens (or the opposite way, start with nothing and add features one by one)
- It is also useful to have **stable explanations**: do not change with small input perturbations (although the model itself may be unstable, in which case your explanations should be at least as stable as the model)
- A way to evaluate fairness: compute faithfulness/stability metrics for datapoints from majority and minority groups, if there is a statistically significant difference, then there is unfairness in the post-hoc explanations
- Just to stress it: one thing is fairness of a model, another thing is fairness of an explanation of a model

### Challenges

- Hyperparameters can significantly impact explanations (random seed, patch size etc.), although that is generally true for all AI models (not just for explanations)
- Since the models and explanations are so different, it's very difficult to compare the approaches
- And even worse, many times, if you run different models to produce explanations, those explanations disagree
- Some methods, when tested more rigorously, showed that the explanations weren't changing when model was changing (basically, take-home point: if your explanations looks very simple and good, and even scores high at faithfulness, you still need to be skeptical and test whether your explanations change when model/model parameters change)
- Post-hoc explanations can be easily manipulated (by doing very small perturbations to the original data)
- You can fool LIME and SHAP to think you are classifying using some feature (e.g. income) while you are actually using another (e.g. only use race to make decisions)
- The explanation using "if-else" rule list, produced for a rule list model may completely obscure all kind of biases that are built into the underlying model
- Sometimes explanations turn out to be counter-productive
- Overall, this calls for more standardization/ more formal methods, benchmarks etc.
- One of the directions is, in addition to an explanation, report the uncertainty of the explanation (confidence interval)
- This approach could also allow user to provide desired confidence interval, and the explanation model to be able to adjust itself to fulfil the requirement
- By 2022, the field is still an area of active research
