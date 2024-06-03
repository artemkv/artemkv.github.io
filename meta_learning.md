# Deep Multi-Task and Meta Learning
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

[Stanford CS330: Deep Multi-Task and Meta Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI)


## TL;DR

- Multi-task learning: learn several related tasks jointly (few tasks, many examples)
- Meta learning: learning to learn a task from few examples (many tasks, few examples)
- The key example to keep in mind is Cezanne-Braque classification of paintings: it does not suddenly make you an expert on Cezanne, but does show how easy you can apply your pre-existing knowledge to perform well on a novel task
- That's because many tasks share some deeper, sometimes latent underlying structure
- If you perfectly learn that underlying structure, all you need to learn for individual tasks is what is truly unique about those tasks (`(a+b) + (a+c) = 2a + (b+c)`)
- That is guaranteed to be less than if you learn the task from scratch in the vacuum (`b < a + b`)
- To use it, you need to model the learning of shared knowledge, i.e. learning how to learn (hence meta-learning)
- So, formulate your task in Cezanne-Braque terms
- In meta-learning, you get the model that is good at learning, not at doing! (You get what you are optimizing for)
- Application: find all the statements that are racially biased in a huge corpus of text. If we had to use a classifier, we would need to label a massive amount of data. However, if we understand English and have some basic intelligence, we could realistically learn the task from seeing just 5-10 examples of racially biased statements
- You still need a lot of tasks, if you don't have it, you can use unsupervised pre-training


## Intro

- Standard approach to RL: learn a **single task** in a **single environment**, starting **from scratch**, using **extensive supervision**
- The robot trained this way is lost when released into the wild, does not translate well to the real world
- This problem is not unique to RL, you have the same situation with machine translation, object recognition etc.
- And while you are training the algorithm on that one specific task, it learns nothing about the rest of the world
- DL is very powerful, as it allows handling unstructured inputs (pixels, text, sensor readings etc.), so you don't have to have domain knowledge or engineer features
- And if you have a large and diverse dataset, that leads to good generalization
- But in many real world applications we don't have large datasets, or we have datasets that are very skewed (i.e. has a long tail of edge cases)
- Humans, on the other hand, can work with a really small number of training examples (e.g. see just 3 paintings of Braque and 3 of Cezanne, and be able to correctly classify a new painting as being one from Braque): **few-shot learning**
- In few-shot learning, you are leveraging the previous experience/knowledge to learn a new task
- The idea is not very new: you can train a visual recognition model to learn many distinct tasks (e.g. classify an animal, but also count animals), and it would be able to re-use the previously learned task for the new one in a relatively few shots
- It was shown in 2019 that, if you train a machine translation model on 102 languages, it actually outperforms a bilingual model (somewhat counter-intuitive)
- This is not always the case: sometimes independent networks really perform better
- Since few-shot learning does not require massive amounts of training data, it helps to democratize DL (anyone can do it)
- Informally, the **task** is: dataset `D` + loss function `L` ⇒ model `f0`
- The **multi-task learning problem**: learn a set of tasks more quickly or more proficiently than learning them independently
- This contrasts **transfer learning problem**: given data on previous task(s), learn a new task more quickly or more proficiently
- Critical assumption to apply multi-task learning: different tasks need to share some structure
- Good news: many tasks do! And even if tasks are seemingly unrelated, underlying laws are the same (e.g. laws of physics)
- Does multi-task learning reduce to single-task learning? (i.e. combine all `Di` and `Li`?) Yes, it can, and it is one of the approaches


## Multi-task learning

- More formally, **task** `i` is the set `{p_i(X), p_i(Y|X), L_i}`, where `p_i(X)` is a distribution on inputs, `p_i(Y|X)` distribution on outputs given inputs, `L_i` is a loss function
- **Multi-task classification**: `L_i` is the same across all tasks
- Example: handwriting recognition, a task corresponds to a single language, but the loss function is the same
- **Multi-label learning**: both `L_i` and `p_i(X)` are the same across all tasks
- Example: face attribute recognition, one task detects the color of the hair, another one the eye color
- The model is a function `f_theta(Y|X, Z_i)`, where `Z_i` is a **Task descriptor**
- Tasks descriptor is something that denotes the tasks that we are interested in (basically, the way to tell the network which task to perform)
- This can be just one-hot vector of size `T`, where `T` is a number of different tasks we want to perform
- This can be something more complicated, like a language description of a task (i.e. a prompt); typically the more information you give the better
- `theta` is all the parameters of a model, a giant vector (just like in stats)
- **Vanilla objective**: `min [sum of L_i(theta, D_i) over all i] wrt theta`, where `D_i` is a training data of a task
- Let's unpack it: look at the error, on a training data, for all tasks, combined, and find the value of theta that minimizes it

### Design choices: conditioning on task

- How to condition on `Z_i`?
- On one extreme, you could share nothing
- For example, if you use one-hot encoding to represent `Z_i`, you could just have `T` different models (e.g. NN) and switch among them based on `Z_i`: formally, `y = sum [1(z_i = j)y_j] over all j`, where `y_i` is an output of each NN
- On another extreme, you could share everything
- Same example, instead of having many NNs, have just one, and concatenate `Z_i` into one of the layers, making all the parameters shared
- So choosing how to condition on `Z_i` is equivalent to choosing how to share model parameters across tasks
- This leads to an alternative view on the MLT-architecture: split `theta` into `theta_shared` and `theta_i` (task-specific), and consider objective `min [sum of L_i({theta_shared, theta_i}, D_i) over all i] wrt theta_shared, theta_1, .., theta_T`
- 3 common choices: concatenation-based, additive conditioning, multiplicative conditioning and multi-head architecture
- **Concatenation-based** conditioning: simply concatenate `Z_i` to the input
- **Additive** conditioning: map `Z_i` into a vector of the same size as an input, and use it as a bias (add the input and `Z_i` together)
- Mathematically the 2 choices end up being the same
- **Multiplicative** conditioning: same as additive conditioning, except you multiply (element-wise) instead of adding (as name suggests)
- **Multi-head architecture**: have a task-specific layers for each task, connected to some shared layers
- There exist more complex choices (cross-stitch networks, multi-task attention network etc.)
- Making a choice is more an art than a science; you kind of have to play with it and see
- Knowing whether sharing would help or hurt is almost impossible
- As a simple rule, if you are overfitting on one of the tasks, share more between tasks

### Design choices: objective

- Should you use the vanilla objective, or should you do something else?
- In many cases you want to weight tasks differently, giving some tasks more weight
- This turns the vanilla objective into `min [sum of w_i*L_i(theta, D_i) over all i] wrt theta`
- You can adjust those weights manually or dynamically, based on various heuristics
- You might decide to optimize for the tasks that is doing the worst, giving it more weight; you would do that iteratively, identifying the worst task at each iteration
- In practice it can become tricky, if you have a large dataset and many tasks, as you would need to calculate the loss on the whole dataset for each task in order to find the worst one

### Design choices: optimization

- How to optimize?
- The approach is, in essence, a stochastic gradient descent:
- Sample mini-batch of tasks (if you have many tasks; if you only have few, just use all of them)
- Sample mini-batch of datapoints for each task
- Compute loss on mini-batch as sum of all errors across all datapoints and all tasks in a mini-batch
- Backpropagate loss to compute gradient
- Apply gradient with your favorite NN optimizer (e.g. Adam)
- Important! Make sure your task labels are on the same scale


## Transfer learning

- Transfer learning is a valid solution for multi-task learning, but not vice versa
- The idea: solve target task `Tb` after solving source tasks(s) `Ta` by **transferring** knowledge learned from `Ta`
- Typically, in this setup, you don't have access to the training dataset you used for task `Ta` when training for task `Tb`
- Example: use pre-trained ImageNet, strip off the last layer, add yours and train it to do medical image classification, using gradient descent
- In this example, we are using weights of pre-trained ImageNet as initial weights, which is an example of learning via **fine-tuning**
- One problem with this approach is that by replacing the last layer with a new one, you are going to slap the layer of random weights on top of your pre-trained network, and, once you start backpropagating, you will be destroying the learned weights of the model and all the knowledge it has
- Simple solution is to freeze the pre-trained layers (and keep frozen or unfreeze gradually); but you can also use smaller learning rate, at least for earlier layers
- Treat those options as hyperparameters
- Transfer learning is still under research, and it's not completely clear what is the best way to do it
- Some papers show that sometimes you don't really need a separate dataset for pre-training; you can pre-train on the fine-tuning dataset. Of course, pre-training in this case should be different from fine-tuning, otherwise you are just doing the same thing, for example, pre-training could be unsupervised, and fine-tuning supervised (not completely clear how)
- Some papers show that, depending on task, it might be better to fine-tune first layer or middle layers, instead of the last layer
- Chelsea's recommended default: start with frozen pre-trained layers, put randomly initialized layer (head) on top, train that last layer until converges, then unfreeze and fine-tune the entire network


## Transfer learning → Meta learning

- With transfer learning, when you train the model for the first task, you don't really consider the second one, you just hope that it helps when you move to it
- **Meta-learning** tries to explicitly optimize for transferability
- That is: given a set of training tasks, how do we optimize not just for learning those tasks, but learning them quickly; so that when we get new tasks, we can learn them quickly too?
- In meta-learning we assume there is a latent information `theta` that is common across all tasks, and a vector of parameters `fi_i` that is different for each task `i`, and the `fi`s causally depend on `theta`
- Let's say you have a family of sinusoidal functions, so `fi` would be a `(phase, amplitude)` pair which you want to learn, `theta` is whatever is common between sinusoidal functions besides phase and the amplitude (the oscillating nature of these functions, kind of)
- If we model this as a Bayesian network, it can be shown that conditioning on `theta` decreases the entropy of distribution of `fi` (`H(p(fi_i|theta)) < H(p(fi_i))`); put in simple words, once you figure out the common structure, there is less to learn for a specific task
- Considering extreme case, entropy `H(p(fi_i|theta) = 0` would mean that `p(fi_i)` collapses to a single value, becoming deterministic; meaning there is nothing else there to learn
- Also, when conditioning on `theta`, all the `fi`s become independent
- So the whole big goal is to recover the common information `theta`, once you have done that, you only need to learn what is truly unique for each task
- Approach: train the model on tasks `T1` ... `Tn`, in a way to solve new task `Ttest` more quickly/proficiently/stably
- Key assumption: meta-training tasks `T1` ... `Tn` and meta-test task `Ttest` should all be drawn from the same task distribution (similar to how in the case of regular ML we assume that test and train data come from the same distribution)


## Meta supervised learning

- **k-shot learning**: learning with `K` examples per class (or total `K` examples in case of regression)
- **N-way classification**: choosing between `N` classes
- Our data for meta learning is a dataset of datasets: `{D_i}`, `i` indexing a task
- Every dataset `{D_i}` is `{(x, y)_j}`, `j` datapoints with input features `x` and labels `y`
- Training dataset `Dtr` is `{(x, y)_1:N*K}`, `K` datapoints for each of `N` classes that we are going to show our model
- The model: function `yts = f_theta(Dtr, xts)`, predict label `yts` on test datapoint, given input features `xts` and `Dtr` with `N*K` examples, leveraging common shared knowledge `theta`
- See black box meta learning section below for a good example
- The meta-learning problem reduces to design and optimization of this function `f`
- General recipe: choose the form of `f_theta(Dtr, xts)`, choose how to optimize `theta` w.r.t. maximum likelihood objective using meta-training data
- The way `theta` appears in the model may not be explicit, and you will not see the exact notation we are discussing here, but on the abstract level, this is what you do
- Real-world application: labeling the data. Labeling is very expensive, so you want to label a very small amount of data and auto-label everything else. Setup: you have a set of labeled satellite images for some parts of the world, but not for others. You could treat every region as a task, manually labels small amount of images from unlabeled regions and run meta-learning algorithm to let the model do the rest of labeling
- _My thought: instead of relying on massive amount of labeled data, you seem to need a massive amount of tasks each containing very small amount of labeled data, if you are not using pre-trained dataset_
- You don't have to give perfect examples to the model to learn. You can train on sketches of cats and dogs, and test on actual photos of cats and dogs. Magic!

### Black box meta learning

- Example: Omniglot dataset, 1623 characters (handwritten) from 50 different alphabets, 20 instances of each character
- Opposite typical ML datasets (MNIST), contains many classes and few examples
- We are going to do 3-way 1-shot learning
- (step 1) Sample 3 characters from a randomly chosen alphabet, this is our task `i` (i.e. learning to recognize these 3 characters)
- (step 2) Sample 2 images per character (one for training and one for testing), assign labels
- The train dataset with 3 images of 3 different characters is an input into the first ("learner") NN, the output is `fi_i`
- Using the images from a test (yes, test) dataset, predict the label using another NN with the parameters `fi_i`
- (step 3) Calculate the error and back propagate into the first networks, making it "learn how to learn" (will not update `fi`s, only the weights of the first network)
- Essentially, instead of learning to predict labels, the network learns how to predict weights that are the best for predicting labels
- Repeat from the step 1, until converges
- In this example Chelsea used RNN for the first NN, the reason is: it's easy to handle data of variable length (You can sample as many characters as you want without changing much of the architecture of this network)
- You can think of slightly different versions of this architecture, but the main idea remains the same
- For example, if `fi_i` turns out to be too big, you can spit out a low-dimensional vector `h_i`, that you can combine with some other parameters to obtain `fi_i`
- Instead of RNN, you could produce embeddings and average them together etc. etc.
- Note that test dataset is not used to evaluate the model, but to evaluate the single task output during training (confusing)
- To evaluate the model, you do the **meta-test**: you pick the task that you haven't seen during training (3 learning examples of 3 classes, 1 testing examples) and pass it to the networks, first one predicting `fi`, the second one using `fi` to predict label
- So again, and the key to understanding this, you are not evaluating the model label predictions, you are evaluating the model's ability to learn from a single example to distinguish among 3 different classes
- Results: 5-way 1-shot Omniglot: 99.07 accuracy, which is quite impressive
- But again and again, we didn't learn to properly classify all characters from 50 different alphabets; we learned to re-use the common structure to learn to distinguish among 3 different characters after seeing a single example for each of them
- So how to use this in practice?
- GPT-3 example: one-shot example is a dictionary definition of a word; the task is to come up with an example of a sentence that uses that word
- Another example: 5-shot examples of poor English vs good English, convert a new sentence in poor English to a sentence in good English
- GPT-3 wasn't actually trained to do meta-learning, the meta-learning has emerged from its architecture
- What makes the few-shot learning to emerge is still under research, but it depends both on model and data

### Optimization-based meta learning

- Is there a better way to predict `fi_i`?
- The key idea is to use fine-tuning (introduced in transfer learning above), by starting with pre-trained parameters `theta`, and calculating `fi` as `theta - alpha*gradient_theta(error(theta, Dtr))`, treating `fi` as a slightly adjusted version of `theta`
- After that, minimize the error on `fi` over all tasks
- `alpha` is a learning rate, in this algorithm, `alpha` is typically larger than `alpha` in a normal supervised learning algorithm
- Usually the pre-trained nets are not that good at few-shot learning, but in this case, the whole algorithm is designed to obtain a pre-trained model that is optimal for this type of learning
- So, compared to black box meta learning, we are eliminating the first NN, but there is still NN that uses `fi_i` as a parameter to make predictions
- Same example of 3-way 1-shot learning, new algorithm:
- (step 0) Randomly initialize meta-parameters `theta`
- (step 1) Sample 3 characters from a randomly chosen alphabet, this is our task `i` (i.e. learning to recognize these 3 characters)
- (step 2) Sample 2 images per character (one for training and one for testing), assign labels
- (step 3) Calculate `fi_i` on every task `i` as described above, `theta - alpha*gradient_theta(error(theta, Dtr))`, `error(theta, Dtr)` is a sum of errors on all the samples from a train dataset (comparing prediction to labels)
- (step 4) Update `theta` as to minimize `[sum of error(fi_i, Dts) over all i]` w.r.t `theta` (using gradient descent), `error(fi_i, Dts)` is an error on a task `i`, a sum of errors on all the samples from a test dataset (comparing prediction to labels)
- So there is an inner and outer gradient descent
- Repeat from the step 1, until converges
- The main difference with the black box version is how we obtain `fi`, the fact that we do that through the gradient descent and not by using a NN
- What we are doing is bringing `theta` to the point that is the closest to all `fi_i` (although there might be no local minima)
- In practice, to compute a full gradient on a step 4 efficiently, you need to do a matrix-vector multiplication, and there is a smart shortcut using Hessian-vector product (probably a topic of a separate course)
- However, this method, still, is typically very compute- and memory- intense due to backpropagation through multiple layers of gradient steps
- To apply the model to a new task, you start with the pre-trained `theta` you obtain, run a gradient descent to fine-tune `theta` into `fi` for that task, and then predict using `fi`
- Yet again, you don't get the model for predicting labels, you get a model that is able to learn to distinguish among 3 different characters after seeing a single example for each of them
- This algorithm is model-agnostic (MAML), it doesn't care how you use `fi` to make predictions (whatever NN architecture you use)
- As for the results, a state of the art 5-way 5-shot models produce ~80% accuracy (in 2022)

### Non-parametric meta learning

- https://towardsdatascience.com/non-parametric-meta-learning-bd391cd31700
- Is there a way to get rid of an inner gradient descent in the optimization-based methods?
- This would save us a lot of complex and costly computations
- Yes, we can replace it with some non-parametric algorithm, such as nearest neighbors
- But what distance metric would we use?
- Idea: learn the distance metric from the meta-training data
- Approach 1: use Siamese network to predict whether two images are the same class
- With 2 images as an input, Siamese networks have a 2-part hidden layer that has the exact same parameters for both images (hence the name), then a common "distance" layer that compares outputs of 2 parts of hidden layers, and, finally, and output layer that spits out the probability of images to be of the same class
- So you could use the output of a Siamese network as a distance metric
- This leads to the following architecture: every image is converted to an embedding, embeddings are passed through the Siamese network: `N` train images for each class and the test image, the Siamese network spits out the similarity, which is treated as a distance in what is essentially nearest neighbor algorithm
- _My note: there is no explicit nearest neighbor algorithm, but really, if you think that the whole "algorithm" is a trivial argmax, it begins to make sense_
- The model is trained end-to-end (using gradient descent)
- However, no gradient descent is required at meta-testing time
- In essence, you are basically learning the best embeddings for the images
- _My note: I feel it gets more and more useful to think about NNs as universal function approximators_
- You can think of some special case situations in which nearest neighbor can fail, example: 2 clouds, one of "+" and one of "-", minuses slightly lower, and some minuses at the bottom of "+" cloud. The test point just below the "+" cloud will be misclassified
- The reason is: we compare the test point to every other point individually
- Instead, we could compute a "prototype +" and a "prototype -", by averaging all "+"s and "-"s, and compare the test point to that
- This is what prototypical networks do
- But then, you of course, start running into more and more special cases when nearest neighbors fail, so it really depends on the task
- The 3 approaches all have pros and cons, but non-parametric one is a good place to start (typically requires the least computing power)
- Unfortunately, it can only be used for classification (not regression)


## Unsupervised pre-training

- What if you don't have access to a large amount of training tasks?
- Task: given a large, diverse, unlabeled dataset `{x_i}`, make a pre-trained model that you could later fine-tune, using small labeled dataset `Dtr_j`, in order to get a good model for a task `j`

### Contrastive learning

- The idea: find representations, such as that similar examples have similar representations
- This could be easy, if you had labels; but if you don't, you could use your knowledge of the world to make assumptions of what should be similar
- For example, if you look at patches of the same image, patches that are nearby can be assumed to be similar
- In the same way, nearby video frames can be assumed to be similar
- Images that you clip and crop can be considered similar to an original image; you could think of many more augmentations to use
- Sometimes your assumptions may break (think of video montage with 2 different scenes pasted together), but as long as this only happens rarely, it's oK
- So you could try to build a function `f_theta(x)` that takes an image `x` and produces a representation, and try to minimize the quadratic difference between representations of similar images `(f_theta(x1) - f_theta(x2))^2`
- If you just do that, you could easily achieve this by producing a constant vector, mapping every single image into the same representation; obviously not what we want
- What we actually want is to **contrast**: i.e. bring together representations of similar images while pushing apart representations of different images
- We can do this in triplets: anchor example `x`, positive example `x+` and negative example `x-`
- Triplet loss: `(f_theta(x)-f_theta(x+))^2 - (f_theta(x)-f_theta(x-))^2`, minimize w.r.t. `theta`
- The term `(f_theta(x)-f_theta(x-)` is unbounded, and in order not to drive it into an infinity, you could use hinge loss (reward difference to a certain point, but once far enough, do not reward anymore)
- Hinge loss is `max(0, loss + epsilon)`, you can only go into negative until `epsilon`, after which 0 will start winning; `epsilon` is a hyperparameter
- A version of this algorithm involves multiple negative examples, using a softmax (the exact loss is omitted here, just look it up)
- This needs a large batch size (ideally, all the negative examples in a whole dataset), but there are techniques that work around that (e.g. momentum contrast)
- Augmentations can be hand-coded, but you could also learn the augmentations in adversarial manner; good augmentations are domain-specific
- Choosing good negative examples is difficult; if the example is too different, you don't really learn from it much ("a table is not an apple")
- Ideally, you want a near miss: a negative example that is very close to a class that we want to learn, but not quite (_My note: back to Patrick Winston's ideas_)
- **Hard negatives mining**: explicitly search for those negatives that are difficult
- _My thought: you don't actually teach kids using examples of "almost horses", so maybe this doesn't have to be too extreme. Yes, they learn what zebra is (also a goat, the Centaurus and a unicorn, but horse with 5 legs is still a horse)_
- SimCLR samples minibatch of unlabeled examples, augments each example twice to get positive examples, negative examples are augmentations of a different image
- Contrastive methods are near state-of-the-art (2022) in self-supervised pre-training for visual data

### Reconstruction-based unsupervised pre-training

- Idea: a good representation of an input is the one that is sufficient to reconstruct it
- This leads to encoder-decoder architecture: `x -> enc -> e -> dec -> x'`, loss is simply `(x - x')^2`
- Once you are done, you could just strip the decoder off, freeze the encoder, put the prediction head on top of it, and fine-tune this head
- As an advantage, comparing to contrastive learning, this does not require positive/negative pairs
- The potential problem with this approach is: if `r` is large enough, it can simply memorize `x` (i.e. if `r` is just `x`), which is not very useful
- Most common approach to tackle this is by having a low-dimensional representations, but this does not always result in good performance
- You need a good bottlenecking mechanism that forces a good low-dimensional representation; in practice `r` often ends up being more like a hash rather than a conceptual summary
- One way to deal with this is by designing more sophisticated bottlenecking (e.g. force zeroes in most dimensions), but that's hard
- Turns out, what helps is to actually make a task "slightly more difficult", common solution is to use masked autoencoder
- BERT is such a model for a text data, MAE for images, can be used as a pre-trained model
- Masked autoencoders are state-of-the-art in pre-training for few-shot learning in language and vision (2022)
- Wouldn't it be nice to have a common architecture that works across modalities? Welcome transformers
- Transformers operate on sequences of data
- To use transformers, you cut the input into pieces, put those into a sequence, for each piece you produce an embedding and concat with another embedding that represents the position in the sequence, then you run a transformer encoding on top
- A transformer encoding is made of Lx blocks (transformer blocks) repeated multiple times (typically, 9-96)
- You are responsible for deciding how to cut the data into pieces, make a sequence out of it, and how to make embeddings, the rest is common
_My thought: that sounds like a bit too much work, I thought in order to be claimed to be generic across different modalities, the model would have to allow to simply throwing any type of data on it_
- _My note: I think transformers are a subject of a whole separate course, so I'm not going to try describing them here_
- Masking is not ideal, since we need to pick a good mask, and you mask a small part of data, so you only learn from that small amount of data. Can this be simplified/improved?
- Welcome autoregressive models (e.g. GPT-3): simply learn the probability of the next token given previous tokens (word, pixel etc.)
- This is analogous to masking every word/pixel; can be seen as a special case of masked auto-encoder


## Meta overfitting (memorization)

- Tasks need to be **mutually exclusive** so that there is no single function that can solve all tasks, otherwise you may see an effect of **memorization** (meta overfitting)
- We want our model `yts = f_theta(Dtr, xts)` to learn to take advantage of both `theta` and `Dtr`
- Memorization can occur when the model starts ignoring one of two: `theta` or `Dts`
- For example, if you always use the same label for the same class on every task, the model may learn to just predict that label from test features `xts` and completely ignore the training set `Dtr`; so just shuffle the labels
- In the same way, if you pass to the model your training set `Dtr` + task description `Z_i`, the model might to learn to rely just on the task description and completely ignore `Dtr`
- Also, model may start ignoring the task description, and start guessing what it needs to do just from the training set `Dtr`
- Solution: modify the optimization to minimize the meta-training loss while minimizing the information in `theta`
- Minimizing meta-training loss will drive model to rely on `theta` more, minimizing the information in `theta` will be an opposing force, and they should balance
- You can achieve this by adding noise to `theta`, so that model can learn to make good predictions with less information in `theta` (meta regularization)


## Unsupervised task construction

- Can you construct tasks automatically from unlabeled data?
- Tasks need to be diverse, in order to cover test tasks well
- Tasks also need to be structured, so that few-shot meta-learning is possible
- You could use unsupervised methods to find image embeddings, run some clustering algorithm to find clusters of embeddings that are found close together in the embedding space, use those clusters as labels and make up tasks that classify original images using those labels
- You can also use domain knowledge to construct tasks: use the augmentations that don't change the label (you can safely flip an image of a dog, but not an image of a character)
- In case of text, you can randomly sample sentences, mask words, and make tasks to predict masked words
- _My thought: comparing to raw images, text has a lot of structure that is readily extractable using non-ML algorithms, e.g. to find the word boundaries you simply search for spaces_


## Latent variable models and variational inference

- This lecture is rip-off of [CS 285: Lecture 18, Variational Inference, Part 1](https://www.youtube.com/watch?v=UTMpM4orS30), [CS 285: Lecture 18, Variational Inference, Part 2](https://www.youtube.com/watch?v=VWb0ZywWpqc), where it is explained much better using the same slides
- The meta-learning methods above can be thought of as point estimates, e.g. they give you `fi` for a given `Dtr`, `theta` (which gives you a distribution of labels)
- But sometimes we want the full distribution, e.g. `p(fi_i|Dtr, theta)`
- These need arises when there is an ambiguity in how to solve a task, and especially in safety-critical context (e.g. medical imaging); but there are other reasons too
- _My note: remember how in Bayesian stats, you get a distribution on a distribution parameter, i.e. distribution on a mean of a distribution_
- To achieve this, we want to build probabilistic models where distributions are more complex than categorical or Gaussian
- In such case, it is usually much simpler to model `p(X)` using latent variables
- Latent variable: some random variable `Z` that we cannot directly observe, distributed `p(Z)`
- Example: we have a bunch of unlabeled datapoints coming from `p(X)`. We may observe there are 3 clusters of data, so we can model `p(X)` as `sum [p(X|Z)p(Z)] over all z`, where `Z` is a latent categorical random variable and `p(X|Z)` is a Gaussian (Gaussian mixture model)
- In general, you have some complex distribution `p(X)`, and some prior for `p(Z)`, typically assumed to be some "easy" distribution (e.g. Categorical or Gaussian)
- You can see `p(X|Z)` as a mapping from `Z` to `X`, this can be done by a NN
- You then assume that `p(X|Z)` is also some simple distribution, e.g. Gaussian (although a mapping itself can be quite complicated, so the parameters of that Gaussian can be very complex)
- This allows us to express `p(X)` as an integral over `[p(X|Z)p(Z)]`
- In general, this allows us to express very complex distributions as products of simple distributions that we can learn how to parametrize
- To train a probabilistic model, you need to maximize this integral w.r.t parameters of these distributions (by maximizing the log of it)
- But these integrals can get really nasty and impossible to calculate (or very computationally expensive to estimate)
- Workaround idea: instead of going through every possible value of `Z`, "guess" the most likely `z` given `x_i`, and pretend it's the right one
- You actually don't guess a single value of `Z`, you guess a distribution `p(Z|x_i)`
- This allows you to replace the integral by `E[Z ~ p(Z|x_i)]*p(x_i, Z)`
- And expectation can be replaced by sampling, so this is nice
- _My note: I'm going to stop here, since it's getting too advanced_

TODO: skipped 10, second half of 11, 12, continue with lecture 13