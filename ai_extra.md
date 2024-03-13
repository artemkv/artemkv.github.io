# AI extra topics

## References

- [Stanford CS 230: Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)


## Intro

- Recent technological advancement produced a lot of data
- Regardless of how much data you throw at a traditional ML algorithms, they seem to plateau eventually, while large NN seem to keep improving
- These 2 factors made DL extremely attractive to researchers and practitioners, which created a positive feedback loop and resulted in exponential advancement in this area
- However, the field of AI is not limited to RL/ML, there exist multiple other tools such as Probabilistic Graphical Models (PGMs), Search, Planning, Learning Representation, Game Theory, and they all keep improving every year
- To be successful in AI, you need to know more than simply how to implement a certain algorithm, you need to understand how to debug it and what to do when the results are not satisfactory
- Leading AI companies tend to have unified data warehouses, instead of disjoint databases, they are very strategic with the data acquisition and leverage automation well


## Full-cycle deep learning projects

- Select a problem → get data → design model → train model → test model → deploy → maintenance
- There are usually loops, as it almost never works the first time
- How do you choose a good project? Which things need to be true for the project to be good?
- Andrew's criteria: your interest in project, data is available, you have a domain knowledge, utility, feasibility
- Try to read some research papers upfront, to see whether this problem has been solved before and what other people are doing when solving similar problems
- Skim through several papers in parallel, do not go into too much details, just get a gist of it; this ensures the best use of your time
- Consider talking to experts and even contacting the authors by email (but do your homework!)
- Do not spend too much time collecting a lot of data. ML projects are iterative, you improve iteratively; if project is novel, you would not be able to predict what is going to be hard about the problem, so you might not even be able to collect the right data
- So spend a couple of days initially collecting data, then try algorithm, then repeat
- Do not rush into data augmentation in the very beginning, until you know it is a good use of your time
- Keep good notes: track what you tried, the hyperparameters you used etc.
- Once you ready to deploy, consider if the algorithm is going to run on edge devices, and what is that device's spec, to avoid running large NN too frequently
- In real world, the data you train the system on is often not the data your system need to perform well on, as data changes (as requirements change)


## Reading research papers

- The field is moving quickly, so to stay on top of the latest state of the art, you need to read papers
- Try to do intentional and systematic, to be more efficient
- Compile the list of papers and articles
- Skim through the list quickly, to get some rough idea, for each paper note the % of how much you read and understood
- Decide which paper/article is actually worth reading in details
- Read that one, then come back to other articles, and see if it is worth reading too
- Repeat
- 5-20 papers should give you a good understanding of the area and implement some algorithm
- Reading 50-100 papers probably means you mastered the subject
- Andrew carries a bunch of papers with him to read at all times
- Do not attempt to read the paper from beginning to the end, instead take multiple passes
- 1st pass: start with title, abstract and key figures
- 2nd pass: intro, conclusions, figures, skim the rest
- Intro and, conclusions are basically what determines whether the paper is going to be published, so the authors usually try their best to summarize the whole paper there
- 3rd pass: read the paper, but skip the math
- 4th pass: read the whole paper, but skip the parts that don't make sense (not unusual!)
- Questions to ask yourself: what did author try to accomplish, what are the key elements of the approach, what could you use yourself, what other references to follow?
- Sources of papers: Twitter (@AndrewYNg), ML subreddit, conferences (NIPS, ICML, ICLR)
- You understand the paper math, when you can re-derive it from scratch
- You understand the algorithm when you can code it from scratch


## Project advice

- Problem description: make sure you describe the problem well (very good abstract)
- Hyperparameter tuning and architecture search: report all what you tried, what worked and what didn't and why
- Writing: make sure to use clear language and no typing
- Explanation of choices and decisions (architecture, loss, metrics, data). You need a good justification for your choices
- Data cleaning and preprocessing (if applicable)
- What would you do next if you had more time
- Interpretation of the results
- Results: achieved accuracy
- No more than 6 pages
- You should be able to pitch your project in <3 minutes


## Career advice

- Preferably, build a T-shaped expertise profile: lots of broad knowledge with some great depth in one particular subject
- Just broad knowledge is preferred to just great depth in one subject
- Better 1 very good project in one area than many tiny projects in many areas. The recruiters are not impressed by the volume of work, but rather by the quality of it
- "Saturday morning problem": no one knows about all the hard work you do over the weekend and there's no short term reward, but if you are consistent for years, you will crush it


## Some example applications

### Match face to id photo

- Run NN on the picture taken by camera, produces a vector
- Run NN on the id photo from the database, produces a vector
- Calculate the distance between two vectors, compare to a threshold
- Train on triplets: 1 **anchor** picture (some person), 1 **positive** picture (another picture of the same person), 1 **negative** picture (picture of a different person)
- The goal is to maximize the distance between anchor and the negative and minimize the distance between anchor and the positive
- The loss is `[dist(ancor-positive)^2 - dist(ancor-negative)^2] + alpha (triplet loss)`
- Alpha pushes network to learn, required if you initialize weights to all zeroes. Without alpha such network would give a perfect loss of zero from the beginning and not learn anything
- Turn this into a face recognition task: use the trained network, run encoding on all pictures, use K-nearest neighbors to find the match
- Turn this into a face clustering task: same but use K-means

### Art generation (style transfer)

- Get encoding (a vector) that represents the content of an image, `content_0`
- Get encoding (a vector) that represents the desired style of an image, `style_0`
- The goal is to produce a new image that has the content of the original image and the given style
- The loss is `[dist(content_1-content_0)^2 + dist(style_1-style_0)^2]`
- We are not training the network to learn the best representations for the content and style vectors, that should be done upfront, using another network
- As a shortcut we could simply use imagenet, and use the 2 first layers to extract the vector representations (there is an algorithm for that)
- Instead, we are training the network to produce the best image out of 2 encodings
- We start with random noise image and extract content and style vectors `content_1`, `style_1` from it
- We compute loss and then decide where to move pixels to minimize the loss (bringing these random pixels closer and closer to the desired result)
- You could also start with content

### Detect trigger word ("OK Google")

- Train on 10 second sounds clips, with positive word marked (where it begins and ends)
- The easiest way to make the clips is to generate them programmatically, using positive, negative words and some background noise
- To make the dataset balanced, you insert as many positive words as negative ones
- You could also try to cut some random voice recordings into 10-second clips and manually label the words in the recordings, but this would be crazily time-consuming, so be smart
- Use RNN (maybe CNN)
- Activation function is a [sequential] sigmoid, at each time step you return 1 after you hear the trigger word
- Loss is a logistic loss, on each time step
- Use **Fourier transform** to convert the sound wave function into the frequency vector
- Talk to experts! Little tweaks in the architecture and the parameters make a huge difference. For example, parameters of Fourier transform
- You could also use triplets, similar to face matching


## Adversarial attacks

- Given a trained network, construct an input that is going to be misclassified
- For example, in case of images, try to construct an image of a dog that gets classified as cat
- The approach is very similar to the style transfer, start with some image of a dog, calculate the loss and decide how to update that image to make the loss go down to get it eventually classified as cat
- The loss should express not only how close the label is to the label "cat", but also, how close the input picture is to the [original] picture of a dog
- What makes attacks possible: space of all possible images > space of images that look real to humans > space of real images
- The space of images that look real to humans but aren't real images is where all the money is (for an attacker)
- **Non-targeted attacks:** simply find an example that fools the AI algorithm (misclassifies, doesn't matter how)
- **Targeted attack:** find an example that fools the AI algorithm into classifying an input as a certain class
- **Black-box attack:** attack without access to model parameters and not being able to back-propagate
- The attacks are often transferrable: if you find an input to fool one network, you may be able to fool another one, using the same input
- How to defend? Same as with security, it is a race, there is no universal good defense, and all you can do is to make it more difficult for the attacker
- One way is to try to detect images that are not real, using another NN, as a pre-check (however, attacker can fool both of networks)
- Another way is to actually generate adversarial examples, correctly label them and train on those (however, this is very costly)
- Yet another solution is to generate adversarial examples while training the network to recognize good examples (also costly)
- Why are adversarial attacks possible?
- The theory used to be that the NN overfit
- In reality, however, the issue is in the linear part of neuron calculation
- Mathematically, you can see that you can find small adjustments of input values that would drive the output to be significantly different
- And actually, in recent years, the NN architectures are pushed to be more and more linear (e.g. replacing sigmoids with ReLU)


## (Generative Adversarial Networks) GANs

- Goal is to train a model to generate the input that looks real, but it is not (e.g. produce an image of a person that does not exist)
- Approach: **discriminator (`D`)** is a binary classifier that detects whether the image is real or generated, **generator (`G`)** is a NN that generates images
- Use the database of real images and train both `G` and `D` together
- `G` will try to fool `D`, `D` will try to beat `G`
- Eventually both should get very good
- If either of them is not good, the image will not look real
- You should make sure to use a good cost function, so that gradients are largest when untrained
- The trick: minimizing `log(1-x) is ~ maximizing log(x) ~ minimizing -log(x)`
- People have actually come up with lots of different cost functions in order to make this process work

### Cycle GANs

- Generate zebras from horses and horses from zebras
- `H` is horse, `Z` is zebra
- `G1` is `H2Z` (horse to zebra), `G1(H)` is a generated zebra
- `D1` is a binary classifier that tells whether the image is real or not
- The forward path: `H → [G1] → G1(H) → [D1] → 1/0`
- Additionally, we have a backward branch: `G1(H) → [G2] → G2(G1(H)) → [D2] → 1/0`
- `G2` is `Z2H`, will try to generate back the input image
- `D2` is another binary classifier that tells whether the image is real or not
- "`G2-D2`" branch ensures that we generate not just any zebra, but the zebra that comes from the original image of a horse
- We also reassemble the same components in the opposite direction
- `Z → [G2] → G2(Z) → [D2] → 1/0` with backward branch `G2(Z) → [G1] → G1(G2(Z)) → D1 → 1/0`
- `D1`, `D2` loss: binary classifier loss
- `G1` loss: average of log of `D1(G1(H))` (the number of times we fooled `D1`)
- `G2` loss: average of log of `D2(G2(Z))` (the number of times we fooled `D2`)
- Additionally, cycle loss: logistic loss between `[H and G2(G1(H))]` and `[Z and G1(G2(Z))]`
- Applications: generate images from sketches, super-resolution (use cycle GANs to convert high to low and low to high, similar to converting horses to zebras)
- Can also be used for anonymizing: generate dataset that looks like medical data set, can be used for training and results in the same parameters, but does not represent any real people


## AI in healthcare, case studies

- There are 4 levels of questions you may ask
- #1, descriptive: what happened?
- #2, diagnostic: why did it happen?
- #3, predictive: what will happen?
- #4, prescriptive: what should we do?
- Example: ECG recording. You get a patch that tracks your heart beat, wear it for 2 weeks, the goal is to detect whether you have an arrhythmia
- With millions of patients recording 2 weeks of data, that's a lot of data, it takes a lot of time to interpret the results
- Approach: label segments of the cardiac rhythm, and map from ECG to sequence of letters, e.g. `[ABBC]`
- Architecture: 1D convolutional NN over time dimension, 34 layers deep, with residual blocks
- _My thought: it's not that trivial, applying DL in real life is hard!_
- Dataset: 30K patients, ~32K minutes, ground truth annotated by clinical ECG expert (very time-consuming)
- When the experts disagree, look at the consensus made by the group, but also their individual assessments
- Sometimes there is no ground truth (sometimes it's impossible for an expert to give a definite answer). In this case you train the model to simply be as good (or better) as an average expert
- Useful takeaway: they plot the heatmap of F1 score (errors between predicted and true label) (matrix labels `X` labels), for model and expert, allows comparing and see where the errors are made, very useful


## Transfer learning

- If you want to use pre-trained network (transfer learning), you need to decide how many layers of pre-trained network you will steal, how many new layers you will slap on top and how many of the re-used layers you are going to freeze


## Unbalanced datasets

- For example, cell segmentation: for every pixel, decide if this is a cell, no cell or a cell boundary
- You are going to have less cell boundary pixels than cell or no cell pixels in your dataset, so the model may be biased towards classifying the pixel as a cell or no cell
- This can be tackled by tweaking the loss function to penalize misclassifying under-represented class higher than other classes


## Interpretability of NN

- How do you interpret the model? Why was certain decision made?
- The question boils down to mapping the output decision back to input space: which part of the input was discriminative for this output
- In case of image classification, which pixels contributed the most to the classification decision? (kind of heatmap for the prediction)
- **Saliency map:** take the score for the given class (e.g. cat) pre-softmax, get the complete expression of how it was calculated, take derivative with respect of input, and this will give the matrix sized the same as input, with every value expressing the degree of contribution of a corresponding pixel to the output score
- You could even try to use this to segment the object
- **Occlusion sensitivity**: occlude a part of the original image (using a gray square), and re-classify the image. See how confident the network is predicting the cat depending on the part of the image occluded. This is much slower than saliency map, especially as you make the gray square smaller and smaller, but much more accurate
- **Class activation map**: at the last conv layer, we are going to get a stack of `N` activation maps, resulting from applying `N` filters. We then apply Global Average Pooling (GAP) layer to convert each of `N` maps into a single number, obtaining a vector of number of size `N`. We pass this vector to a fully connected layer to produce a label. Having done that, we then find how much each feature map contributed to the label and produce the weighted sum of all the feature maps, which we can then map back to the input to highlight the right areas
- To find how much each feature map contributed to the label we basically calculate the score for each cell of each feature map
- The actual math is a bit tricky, but the important thing is that we don't need a gradient to calculate it, we just re-arrange the GAP and the score calculations in a smart way
- If you do this, you can see where the network is looking at on the picture, to make the prediction

### Visualizing the inner layers

- With the previous methods, you can see that NN understands where the cat it, but what does it mean, from the point of view of NN, to be a cat?
- **Class model visualization**: do the backpropagation of a class label all the way back to the input, and to use the gradient ascent to generate an image that maximizes the score of a cat
- This will make NN dream of (many) cats
- **Dataset search**: pick a feature map, and run the NN across the whole dataset to find top 5 images that produce the maximum activation of that particular feature map. Map the highest activated cell of the feature map to the corresponding area of the original image to see the cropped piece that produced that activation
- **Deconvolution**: kind of encoder-decoder architecture, to go from the label back into the original dimensions, reconstructing the zone of influence (for a particular activation map) in the input space
- The whole conv net gets mirrored back: max pool converts to unpool, conv to unconv etc.
- The tricky part is to represent deconv as a matrix and vector multiplication, to do the calculation efficiently. Fortunately some smart people did that
- The conv operation can be presented as a vector (of conv inputs) by matrix `W` multiplication, which allows us to derive deconv as an inverted matrix `W` by vector (of conv outputs) multiplication. Technically, this would require `W` to be invertible and orthogonal, which is not always true, but we would ignore it, hoping that it is going to produce some useful result nevertheless
- Turns out, to do deconv in 1 dimension, you can take your original conv matrix `W`, flip the weights, modify the output vector (divide the stride by 0, insert zeroes) and multiply. The similar kind of trickery goes into the implementation of 3d deconv
- Deconvolution is quite complex and there is a couple of other hacks that go into the implementation, but it allows you to reconstruct all the activation maps from all layers
- The results are actually quite impressive, you can see that the first layer filters are looking at the edges at different angles, the next layer filters detect more and more complex shapes
- Essentially, NN learns whatever is required to make a final prediction
- What is interesting is that the final label is all what we provide the NN with in order to force it to learn
- _My thought: so maybe there should be much more research into how to construct a good cost function, that matches the real human learning_
- **Deep dream:** run a network, predict a label, and run it back to drive a certain label up. This makes everything that looks like a cat to actually turn into a cat


## Q-learning

- This relates to application of DL in RL
- The naive approach would be, given state, try to predict the next move; but this has some huge issues
- The trouble is, in many applications of DL, the number of possible states is simply unmanageable. For AlphaGo, there are 10^170 possible states, more than there are atoms in the universe
- There is often no single ground truth. In AlphaGo, you may ask experts what would be their next move, but different experts would give different answers (maybe depending on the strategy they follow or their personal style)
- So the model would likely not generalize
- So we need to re-frame the problem in the shape of RL, and try to use DL in that context
- Q-table is the matrix in size `#actions X #states`, that, for each state `s` and each action `a` taken in that state, gives you the value of a long-term accumulated reward, denoted as `Q*(s, a)` (optimal Q function)
- The value of a state depends completely on the policy; in other words, the value of a state is equivalent to the value of the highest action taken in that state
- `Q*(s, a) = r + gamma*max(Q*(s', a'))`, a classical Bellman optimality equation
- This means the total cumulative reward is sum of immediate reward `r` and the maximum possible total reward at the next state `s'` over all possible next action `a'` that can be taken in state `s'`, with a discount factor gamma
- _My note: so nothing new comparing to what I already learned in RL. What is a bit different is that this model seems to be strictly deterministic, as we always choose an action that maximizes reward instead of sampling from distribution, and the state transition is guaranteed to bring us to state `s'`. As a result, Bellman equation is expressed in terms of max over `Q*`, not in a form of expectation_
- If we calculated the Q-table, we could, at any state `s`, decide which action to take, by just picking the one with the max reward associated to it
- Trouble is, again, with huge number of states, it is not feasible to construct the complete Q-table
- This is why we want to approximate the function `Q*(s, a)`, i.e. the Q-table by the NN
- This is where Q-learning becomes Deep Q-learning
- NN are very good function approximators, that's basically what they are
- Input: the current state `s`, so in the size of state vector (for AlphaGo, `19*19*3`)
- Output: for every possible action `a`, `Q*(s, a)`, so in the size of number of actions (one output neuron for every possible action)
- Loss function: `L = (y - Q(s, a))^2`, `y` is the target value, taken action `a`, `Q(s, a)` is the prediction of the network (this is different for every individual action `a`)
- Note that we cannot use the logistic loss, since our output is not a class, but a total reward. We can have exactly the same value of reward for each of the actions
- But how do we calculate the target value `y`???
- Well, it's kind of easy, right? `y = r + gamma*max(Q*(s_next, a')) over all a'` (`r` is immediate reward taking that action)
- But wait: it's recursive, and if you don't know how to calculate `Q*(s, a)`, you have the same problem for `Q*(s_next, a')`
- Solution: just use the current state of the NN to predict `Q*(s_next, a')`, and, based on that, calculate `y`
- So, essentially, use the same NN to predict both the value and the target
- This is, of course, not correct (and kind of crazy), but it can be shown to converge to the right value
- A problem: when computing a gradient, `Q*(s_next, a')` will be non-zero and constantly updating as the NN updates, so it makes the target constantly moving which is bad for training
- To avoid this, the network predicting the `Q*(s_next, a')` should be fixed for a batch of updates (100K or 1M)
- Algorithm: initialize the NN with some initial weights, then loop over episodes. For each episode, start from state `s`, and loop over timesteps, until the end of the episode. On every time step, use the NN to predict total reward for all possible actions. Execute the one with the highest predicted total value, observe immediate reward `r` and next state `s'`. Compute targets y by forward-propagating the `s'`, now you have everything to compute loss. Update parameters using gradient descent
- There are several tweaks that go into the real implementation
- First, handle terminal states correctly
- Second, when training atari breakout, there is a lot of correlation between the sequential states, those states are often of low significance, and the important states are rare (like when the ball bounces against the wall). Essentially, this is class unbalance issue, and you need to tackle that
- Third, if we always pick the best current action, we always exploit and never explore. So instead you should sometimes take a random action (controlled by a hyperparameter)
- Deep RL is good at many atari games, but all the NN sees is pixels, while human can extract much more context from the image. We see the key, and we understand it can open the door. So games that rely more on that extra knowledge and that are very long are still hard for the AI to beat


## Chatbots

- Goal is to detect the intent and generate an appropriate action and response
- Intent can be, for example, "enroll" (to enroll in the course), or "inform" (to request some information about the course)
- If the intent is "enroll", the chatbot would enroll the user and say something like "you are enrolled"
- The chatbots work the best when they are really targeted (very few intents)
- Every intent requires some arguments, i.e. some information from the user: slots
- Example of slots are: name of the course, year, quarter etc.
- Once we know the intent and all the necessary pieces of the information, we can just call an API to get the data
- So normally you need a knowledge base and the API, matching intents you support
