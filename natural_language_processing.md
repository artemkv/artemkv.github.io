# Natural Languages Processing

## References

- [Stanford CS224N: Natural Language Processing with Deep Learning](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)


## Intro

- Levels of analysis: Raw text → Tokenization → Morphological analysis → Syntactic analysis → Semantic analysis → Discourse processing
- Representation learning attempts to automatically learn useful features
- Natural language is hard, because humans leave a lot of common knowledge out of the communication
- There is a lot of structural ambiguity: "The Pope's baby steps on gays", "scientists study whales from space", there is a lot of knowledge about the world needed to understand these sentences correctly

### Application

- Spell checking, keyword search
- Extract information from websites: product price, dates, location, company name
- Classifying: detect reading level of school text, positive/negative sentiment
- Machine translation, complex question answering, automating customer support

### Deep learning approach

- Represent meaning of a word as a vector of real numbers (usually 25-300 dimensional vectors, can go up to thousands)
- Humans are not good with multidimensional spaces, so to visualize them you would usually project the vectors on just 2 or 3 dimensions (PCA, T-SNE); this can be very misleading
- As words can be split into morphemes (`"un"-"interest"-"ed"`), every morpheme can be represented by a separate vector, and we can combine multiple vectors in one
- NNs can be used to detect word dependencies and determine the structure of sentences (syntax analysis)
- Semantics: every word and every phrase and every logical expression is also a vector, NN combines 2 vectors into 1 vector
- WordNet is free taxonomy of English language (is-a relationships, synonyms etc.)
- WordNet is hand-made, so it's very difficult to keep up-to-date, and has more problems


## Word2vec

- One way to convert words to vectors is to consider words to be atomic symbols
- This leads to **one-hot encoding**
- The length of the vector is the total number of words in your dictionary, for each word, you have 1 in one position and 0 in all other positions, e.g. `[0, 0, 0 ... 1 ... 0, 0, 0]`
- This representation is problematic, as it fails to capture any kind of relation between words, there is no inherent notion of similarity
- Such vectors are **orthogonal**, i.e. the dot product of the two vectors is zero, so we cannot do anything interesting mathematically
- Example: "hotel" and "motel" would be 2 completely unrelated vectors, just like "stick" and "herring"
- Also, the size of the vector is bound by the size of a dictionary, and for natural language it is a huge number
- This is, however, what people did until about 2012
- We could deal with similarity separately, for example, by building a table with numbers that, for each combination of words, represent similarity (Google did it around 2005)
- But instead we prefer the direct approach where vectors themselves encode it
- _My thought: you could probably create new words like this, or predict some words that are not in the dictionary_
- This approach would allow us to calculate similarity of complete sentences by calculating inner product
- You might also expect to do some math like "Germany" + "capital" = "Berlin"
- Homonyms are packed into the same vector, which is an inherited weakness, and there were attempts to battle it
- The approach that is currently used is based on the concept of **distributional similarity**: represent a word by means of its neighbors
- The idea goes back to 1957, J.R. Firth: "you shall know a word by the company it keeps"
- Word2vec is introduced at Google 2013, and it is built around this idea
- As for many other deep learning models, you need a large corpus of text
- Word2vec is not a singular algorithm, rather, it is a family of model architectures and optimizations that can be used to learn word embeddings from large datasets
- 2 main algorithms: **Skip-grams (SG)** and **Continuous Bag of Words (CBOW)**
- **Skip-grams (SG)** predicts words within a certain range before and after the current word in the same sentence
- **Continuous Bag of Words (CBOW)** predicts the middle word based on surrounding context words
- Simple rule of thumb: Skip-grams (SG) is better, but more expensive
- 2 moderately efficient training methods: **Hierarchical softmax** and **Negative sampling**

### Skip-grams (SG)

- The goal is to come up with word vector representations (embeddings) that are the most useful in predicting words within a certain range before and after the current word in a sentence (the context)
- How do we know if the particular vector representation is useful in predicting the context? Well, we actually need to try to use them for that purpose and see
- So we need to build a model that, given a word and word vector representations, would predict the context, and optimize this model by adjusting the vector representations of words in order to get the best predictions
- As an example, take the sentence "...turning into banking crises as..."
- Assign every word an index, from 1 to `T`
- For each word in position `t` from `[1, T]` (**central word** or **target word**), consider surrounding words (or **context words**) in a window of radius `m`
- Given the window of radius `m=2`, let's consider the word "banking" as a central word in position `t`
- The context words will be: "turning" at `t-2`, "into" at `t-1`, "crises" at `t+1`, and "as" at `t+2`
- We don't actually care about the particular position of a context word inside the window
- Our model will accept a word as an input (one-hot encoded), convert it to a vector representation, and produce **prediction vector**
- The prediction vector would contain, for each word in the vocabulary, the probability of that word to appear in the context of the input word, within the radius `m`
- Suppose that, using our model, we predict that `p("turning"|"banking")=0.5`, `p("into"|"banking")=0.2`, `p("crises"|"banking")=0.7` and `p("as"|"banking")=0.1`
- Now we can estimate how good these predictions are
- Since we actually see these 4 words in our sentence, we would like our model to predict exactly these 4 words, so the predictions for these words should be as close to 1 as possible
- The same logic applies to every position `t`, so we are going to optimize these numbers across the whole sentence
- Considering every word in a sentence, our total loss function will be: `J = product of p(Wt+j|Wt), for all t from 1 to T, for all positions -m to m, excluding j`
- To make this loss function independent of `T`, we divide by `T`
- This product should be as close to 1 as possible, but for simplicity, we can also just maximize this quantity
- As usual, instead of maximizing probability, you would maximize the log of probability
- In machine learning, the convention is to minimize the cost function, so we will minimize negative log probability
- `m` is a hyperparameter, can also be optimized
- Once the model is trained, we don't need it anymore: we are not going to actually predict the context words
- There are high frequency words like "and", "or", and they distort the vector representation
- There are techniques to tackle that

### Skip-grams (SG) implementation details

- https://aegis4048.github.io/demystifying_neural_network_in_skip_gram_language_modeling
- http://ronxin.github.io/wevi/
- Take the corpus of text and produce all possible pairs of `[central word, context word]`, then use it for training of the model
- `V` is the size of the dictionary
- `d` is the number of dimensions of a vector representation
- For simplicity, people usually use 2 different vector representations: one for a central word and one for a context word (the latter is discarded in the end)
- So we will have `2d*V` parameters
- Start with some matrix `W`, sized `dxV`, with initial word vector representations for the central word
- We also need a matrix `U` for context word vector representations, sized `Vxd`, with some initial values
- Initial values for both `W` and `U` are "random", can literally be random, but normally you would apply some best practices for NN initialization
- For each pair [central word, context word]:
- 1). Convert the input central word into one-hot encoding `Wt`, and pass to the model
- 2). Multiply `Wt` by `W`, effectively selecting the corresponding central word vector representation `Vc`
- 3). Multiply by context word matrix `U`, to obtain output prediction vector for the context word (any position in the context)
- 4). Apply softmax, to obtain probabilities (in practice is too expensive so is replaced by some cheaper functions)
- `Softmax(xi) = exp(xi)/(sum of exp(xj) for all j)`
- 5). Calculate the loss function (prediction error)
- 6). Update parameters (vector representations) using gradient descent
- Since this would be hugely expensive to update all parameters for all training samples (batch gradient descent), usually parameters are updated for every training sample (stochastic gradient descent) or in mini-batches

### Negative sampling

- Idea: train binary logistic regressions for a true pair (central word + word in context window) versus several noise pairs (central word + random word)
- Basically, instead of using true probabilities applying softmax across all vocabulary, use sigmoid and produce "true" for one of the words and "false" for some others, randomly selected
- Maximize probability of context words producing "true" comparing to random words
- When sampling random words, try to sample less frequent words more often (there is a formula to make this unigram distribution that you can use for pulling your sample from)
- Negative sampling allows to greatly reduce the amount of calculations required to update the weights

### Counting

- Why not capture co-occurence directly? Just make a table `V*V` and count words in the neighborhood, using the same window `m`
- Once you're done, you could even use the table directly, the rows (or columns) being vectors
- One problem is the high dimensionality of the table
- Another problem is the sparsity, which is usually a bad thing in ML
- Both problems can be tackled using standard dimensionality reduction techniques
- People actually did this in the past, but it never worked very well
- However, there are ways to optimize, such as adjust counts for frequencies, count closer words more etc. which produces some interesting results
- Currently, the 2 approaches coexist and both have some advantages and disadvantages

### GloVe

- Word2vec, in essence, tries to capture word meaning by looking at word co-occurences
- Idea: why don't we calculate the ratio of co-occurence directly, over entire corpus?
- We'll go through the entire corpus and fill the co-occurence matrix
- Similar to Word2vec, we will only consider words in the direct proximity, using a window of a small size (usually 5-10)
- The vectors that come out of co-occurence matrix are very huge (since the matrix is `V*V`), so we need to reduce them to 25-1000 dimensions
- One way to do it is to apply SVD (Singular Value Decomposition) to the co-occurence matrix (`numpy` can do it)
- There are different heuristic adjustments to achieve better representations (e.g. ignore most frequent word)
- There are several disadvantages of this method
- First, SVD gets bad when you get to millions of words
- Second, it turns out, word representations that come out of SVD perform less well downstream, comparing to Word2vec
- Stanford's GloVe algorithm tries to get the best of both worlds
- GloVe starts by calculating the co-occurence matrix
- Similar to Word2vec, GloVe then creates 2 matrices `W` and `U`, and learns the right values for those, using co-occurence matrix
- However, instead of going through the original corpus, GloVe goes through the co-occurence matrix
- For each pair of words, it tries to minimize the difference between `UiWj` and the `log(Pij)`, where `Pij` comes from the co-occurence matrix
- So co-occurence matrix gives us proper statistical data, but instead of squashing it using SVD, we get to the vector representation through training
- _My thought: all these methods are trying to learn language in the vacuum, with no connection with the real world_
- _My thought: the closeness of 2 words in the sentence is also completely stupid metric. This is not how the languages work at all: the words that are the most related are not necessary the closest in the sentence. Think Dutch where the verb goes all the way to the end_
- _My thought: there are even some meta connections, like when you refer to the previously stated fact by "this" or "that", or "en"/"y", something that even little kid would understand but what is completely thrown away_
- Most of the tests are done against Wikipedia, mostly English, but Wikipedia is not really representative of all possible text
- Regardless of how the vectors are constructed, and even with all the flaws of the algorithms, they are shown to be more powerful than just random vectors
- _My thought: maybe this is the biggest insight of the whole idea: the language is not the combination of symbols, but the combination of ideas_
- According to 2013 evaluation, GloVe beats Word2vec

### Evaluating word vectors

- Intrinsic: evaluation on a specific/intermediate subtask
- For example, estimate similarity between 2 words, on the scale 1 to 10, using the vectors and compare to human judgement, see how well they correlate
- Today, more popular method is vector analogies ("man" is to "woman" like "king" is to ?, "sushi" - "Japan" + "Germany" = "bratwurst")
- There are databases of analogies that you could use to evaluate your results
- The performance on the intermediate task should correlate with performance on a real problem, otherwise useless
- Extrinsic: evaluation on a real task (e.g. entity recognition, like identifying person or location)
- The problem is it can take long time (imagine it takes weeks to train the final model)
- Also, it might be not clear if the problem is really the vector representation or something else
- Not every vector representation is good for every downstream task
- For example, word similarity based on co-occurence is not really useful for sentiment analysis, since words like "good" and "bad" tend to appear in a very similar context and get quite close in vector space
- So for the task of sentiment analysis, using random vectors can actually do better!

### Hyperparameters

- Number of dimensions
- Larger word vectors don't necessarily perform better (pick 300, not 1000)
- Windows size
- Asymmetric context (only words to the left/to the right)
- Some studies suggest that choosing correct hyperparameters might be more important than choosing between NN-based or count-based model

### Tricks

- Instead of using words as inputs, you can split words into subwords: "company" → "com" + "pa" + "ny", and find vector representations for those
- This dramatically reduces the vocabulary size


## Word Window Classification

- Single word classification is rarely done, since the word meaning hugely depends on the context
- So usually we deal with the task of word classification in its context window of neighboring words
- Example is entity recognition, which groups words into 4 classes: "person", "location", "organization" or "none"
- Example: center word "Paris" in the context window "museums in Paris are amazing", should recognize "Paris" as "location"
- We take all the words in the window (5), find their vector representations and stick them together into a 5d matrix
- The simplest classifier would simply add softmax on top of that matrix (and then train, using stochastic gradient descent)
- As a result, you update your word vectors, to maximize predictions
- To gain accuracy, you can add an additional neural layer before the softmax
- _My thought: so this is just a basic classifier, the main insight is how you define the input_
- Instead of softmax, you could use max margin loss
- While softmax simply focuses on "drawing" a correct decision boundary, max margin loss tries also to maximize the distance from the decision boundary, creating more robust decision boundary


## Dependency Parsing

- Human languages have structure, sentences are put together according to rules of grammar
- **Phrase structure grammar** (aka constituency grammar or context-free grammar) is a formal grammar which is used to generate all possible strings in a given formal language
- For example, "verb comes after noun", "adjective may appear before noun"
- The term was introduced by Noam Chomsky, who believed in universal grammar
- The beauty of this approach is that you can describe every possible sentence of a language using simple finite grammar
- Another way to structure the sentence is by identifying dependencies between words (dependency grammar)
- For example, in the sentence "large barking dog by the door", the word "barking" depends on the word "dog", as its purpose is to add some extra meaning to the word "dog"
- So instead of thinking about the common underlying structure of a languages "[adjective] → noun → [adverb] → verb", you think about how the specific words depend on each other: "dog ← barking"
- Dependency grammar can help to resolve ambiguity in cases when constituency grammar fails: "Mutilated body washes up on Rio beach to be used for Olympic beach volleyball"
- When dealing with prepositional sentences, it can be very tricky to understand which part of the sentence depends on which
- Example: "The board approved its acquisition by Royal Trustco Ltd. of Toronto for 27$ a share at its monthly meeting"
- There exist databases of universal dependencies across multiple languages: https://universaldependencies.org/
- These databases are manually maintained, building one is much slower and costly than a constituency grammar
- However, these databases capture the meaning behind words, so there is a connection to real world
- In particular, these databases are invaluable for evaluation, as they provide ground truth, golden-standard data
- Unlike in linguistics, in natural language processing, dependency grammars dominate, as they show to be much more useful (sorry Chomsky), and the trend is ever-inreasing
- Also, dependency grammars seem to work better in languages where the word order is less strict comparing to English (e.g. Russian)

### Dependency grammars

- In dependency grammars, the syntactic structure consists of words and (one-way) dependencies (arrows) between words
- The arrow connects **head** to a **dependent** (some people draw the arrows in the opposite direction)
- One curiosity: prepositions do not have dependents, they are treated like case markers
- Every dependency has a type (e.g. subject, apposition etc.)
- One has to make decision what is head and what is dependent, and on the type of dependency
- What can help is looking at the distance between words
- Also, you may identify words that tend to have more dependencies than others (e.g. verbs)
- Every sentence translates to a graph that has a single head, is acyclic and connected
- The sentence is parsed by choosing for each word what other word is it a dependent of, thus creating a tree
- Common ways to do dependency parsing is using dynamic programming or graph algorithms, but one of the most popular methods today is **deterministic dependency parsing**
- Deterministic dependency parsing is greedy algorithm guided by ML classifier

### Arc-standard transition-based parser

- Let's parse "I ate fish"
- Start with a `[root][I, ate, fish]`, first array being a **stack**, second one being a **buffer**
- Stack's top is to the right
- Buffer's top is to the left
- There are 3 (very simple) operations that we can perform: `shift`, `left-arc`, and `right-arc`
- **Shift:** take the work at the top of the buffer and put it at the top of the stack, `[root][I, ate, fish] → [root, I][ate, fish]`
- Second shift: `[root, I][ate, fish] → [root, I, ate][fish]`
- **Left-arc**, **right-arc**: make attachment decision by creating a dependency arrow to the left/right, and drop the word from the stack
- `Left-arc`: `[root, I, ate][fish] → [root, ate][fish]`; on the side, remember arrow "ate" → "I"
- `Shift` again: `[root, ate][fish] → [root, ate, fish][]`
- `Right-arc`: `[root, ate, fish][] → [root, ate][]`; on the side, remember arrow "ate" → "fish"
- `Right-arc` again: [root, ate][] → [root][]; on the side, remember arrow "root" → "ate"
- Which operation to perform is predicted by a discriminative classifier (SVM is a good choice)
- Since we have a database of dependencies, you can easily train the classifier on it
- If you want to add dependency labels, you just extend your set of operations ("left-arc as an object", not just "left-arc") and build a classifier that can predict those classes
- The beauty of the algorithm is that it gives you extremely fast, linear time parser
- You can improve the algorithm by making it explore different alternatives, instead of greedy picking the next operation
- Most of the time, you get marginal improvement in accuracy while paying significant price in performance
- Accuracy can be measured by just looking at the arrows (UAS score), by just looking at the labels (LAS score)
- Of course, you need to decide which features to use to train the classifier
- If you just look at word representations, this would be a very sparse feature
- So in addition to that, you could come up with some **indicator features**: which word is second on the stack, which tense the word in etc.
- However, it turned out computing these features would account for 95% of the time consumed by computation
- Alternative is: use d-dimensional dense vectors for words, and use the same approach to learn d-dimensional vector representations for parts of speech and dependency labels, concat them all together and use as a feature
- The input of a classifier is the stack + buffer, one hidden layer, softmax output layer
- This approach was used by Chen and Manning in 2014, and resulted in the first simple, successful neural dependency parser, outperformed greedy parsers in both accuracy and speed
- SyntaxNet is based on Chen and Manning with some improvements


## RNN language models

- Example problem: compute a probability for a sequence of words
- This can be useful in machine translation, to decide between 2 potentially correct versions of a sentence
- For example, deciding between 2 different word orders: "the cat is small" vs "small is the cat"
- Or deciding between 2 different possible translations: "walking home after school" vs "walking house after school"
- In traditional language models, we condition this probability on window on n previous words, and estimate probabilities from frequencies, but this requires huge amount of memory to store counts
- Modern approach uses RNNs
- So if you want to predict the next word in a sentence, you just run the whole corpus through the RNN to train it
- Classical problem with RNN is vanishing/exploding gradients
- Vanishing gradients prevent words from time steps far away to be taken into consideration

### Application

- Entity recognition
- Opinion mining: classify words as **direct subjective expressions** (explicit mentions of private states) vs **expressive subjective expressions** (expression that indicate emotions without explicitly conveying them)
- Example: "The committee, as usual, has refused to make any statements"
- "has refused to make any statements" is DSE
- "as usual" is ESE
- The authors wanted to find the beginning and the end of ESEs/DSEs, so they modified their RNN to be bidirectional
- You just go through the corpus once left to right, and one right to left


## Machine translation

### Traditional approach

- Traditional approach, `e` is English, `f` is French, translate `e→f`
- Determine `P(f|e)`, `P(e)`, then find the most probable arrangement of translated words
- Train model on parallel texts
- To know `P(f|e)`, you need to know alignments of words in 2 parallel texts, which is a very hard problem in itself already
- Alignments can be 0:1, 1:0, N:1, 1:N and N:N
- Different languages have completely different order of words
- One way you can determine the alignment is by applying grammar rules, and matching on parts of speech
- But let's assume you managed to get all the possible alignments
- Now you get all the possible translations for each of the words
- At the end, you want to combine the pieces in a grammatically correct form
- This produces hugely complicated systems that require a lot of work and usually produce crap

### Deep learning approach

- First approximation: encoder - decoder architecture
- Encoder part is RNN, going through the sentence, word by word, until the end
- Decoder part is the same RNN that stops taking the input and only listens to the previous step, and uses softmax to generate next word in the target language, until generates some "stop" word
- First modification: use different weights for encoder and decoder
- Second modification: decoder's input is not only the previous hidden state (like in any RNN), but also the last hidden vector of encoder + previous predicted output word
- More modifications: deep RNNs with multiple layers, train the network in both directions
- It is also a good idea to pass the input sequence in the reverse order, so instead of training on `ABC→XY`, you train it on `CBA→XY`, this way the last word processed by encoder will be the first word to to begin translation with
- You can also decide to feed the encoder result into each step of decoder, together with the result of the previous step in decoder
- Final improvement: use better units (GRU/LSTM), potentially stacked together
- LSTM are very powerful and are default model for most sequence labeling tasks, require a lot of data
- By 2016 deep LSTMs killed traditional machine translation systems
- In fact, the results are pretty incredible
- You can train ensemble of models and pick the most common suggestion, improving results even more
- The same architecture can be used to generate chatbots: basically, instead of generating translation, the model would generate a response

### Advantages

- End-to-end training: all parameters are optimized simultaneously
- Exploit word similarities (Word2vec instead of 1-hot encoding)
- Better exploitation of context: both source and partial target text
- Everything is done by one system, no need for many "black box" components for different things like reordering, transliteration
- The model is so compact, you can run it on your phone

### Pitfalls

- The model described above would never be able to predict the word it hasn't seen before
- To battle this, combine softmax with "pointer" to one of the words from the input (Pointer Sentinel - LSTM model)
- The model may overlook some clear grammatical markers
- Default encoder-decoder model can only handle one language pair
- However, it turns out, if you modify your input by adding an artificial token that indicates the (target) language, you could train the same model at many language pairs
- Such model would not only generalize across languages, but even be able to handle language pairs it has never seen before!


## Models with Attention

- In an original encoder-decoder architecture, the decoder is fed with the last result from the encoder (and the previous hidden state)
- So the whole input sentence is captured in that one single state
- This works well on short sentences, but the performance quickly drops on longer ones (>30 words)
- The idea is, instead of using that single final state of the encoder, store all the source states (all the intermediate outputs of the encoder) in a pool, with random access, to be used by the decoder
- In the decoder, as you are trying to generate next word, decide which source state to look at, and use that state as an input (together with previous hidden state)
- Decide which source state to look at == where to "pay attention"
- The idea has an intuitive interpretation: essentially, you are making links between words in the source and target sentence
- Use previous hidden state to decide which source state to look at
- For that, calculate a score between previous hidden state and each of the source states, normalize using softmax, and combine all the source states weighted by the score
- Feed that into the model, together with the previous hidden state, to predict the next word
- To calculate the score, you need an **attention function**
- Attention function can be an actual function, build based on some kind of heuristic, can be a simple single layer NN, can be 2-layer NN etc.
- One modification of this approach is using "local attention", i.e. using only a subset of all source states when predicting the next word
- The models with attention turn out to work better even on short sentences


## CNN

- RNN is sequential, which can strike back
- Often capture too much of last words in final vector
- Directional, relies of prefix context (context on the left so far)
- The idea of CNN: instead of computing a single representational vector at any time step, depending on the previous words so far, compute a phrase vector for every single phrase
- You would start with small convolutions and build it up
- Example, in the sentence "the country of my birth", start with vector representations for a pair of words: "the country", "country of", "of my" etc., then move to 3-words vector representations etc., until you have a single vector representation for the whole sentence
- Start with word vectors, use Word2vec or similar
- Concatenate all the word vectors to get the vector for a sentence
- Choose the size of the convolution (number of words) and slide over the sentence
- The mechanics is the same as in a normal CNN: you zero-pad, max-pool, use multiple stacked convolutional filters etc.
- Max pooling allows to give an extra importance to a word "not" for a filter (as an example)
- At the final level, you could use softmax (across the stack of convolutions), if you want classification
- You could use standard tricks as dropout
- Dropout prevents overfitting to seeing specific feature constellations


## Subword Models

- Parts of words often have their own meaning, and modify the meaning of the whole word: `"unfortunately" = "un" + "fortun(e)" + "ate" + "ly"`
- There were attempts to build models that properly split words in parts and encode them separately, but it is a complicated task
- Instead, we could just go directly to the character-level processing, and in practice those models perform similarly to the models that do the split "properly"
- You could represent the word as a set of n-character n-grams: "hello" would be ("#he", "ell", "llo", "lo#")
- There are extra advantages to this approach
- There is infinite space of possible words, and you need to deal with the huge vocabulary, but there is a limited set of character trigrams
- Also, in a social media world, many words are not written in a canonical form: "Goooooooood!!!", "imma go, u want something?"
- There can be 2 ways to apply this approach
- First, you could use character-level embeddings to create word embeddings, and then continue working at the word level
- You can run convolution over characters in words to generate word embeddings
- You can run RNN over characters in words to generate word embeddings
- Second, you could completely forget about words and do all the processing at the character level
- Surprisingly, both methods turn out to be very successful
- You can build a successful machine translation system working purely on character level
- Intuitively, the reason it works well is that in many languages (Czech, Russian), the parts of words capture a lot of information (and not prepositions)
- Disadvantage of working purely on character level is your sequences grow much longer
- You can use compression algorithms to clump some characters together ("byte-pair encoding")
- You start with the vocabulary that consists only of letters of a language, "a", "b", "c", "d", etc.
- Then you find the combinations of letters that are frequently seen and add them to the vocabulary, starting from 2-letter combinations, and going up
- For example, if you detect that "e" is often followed by "s", so you add "es" to the vocabulary
- This way you can decide upfront on your max vocabulary size
- This idea works really well in practice
- Hybrid models are also used, to combine best results from both approaches


## Tree-structured Neural Networks and Constituency Parsing

- This is yet another model for natural text processing
- How can you go beyond words and do semantic composition?
- For example, "the country of my birth" is 5 simple words, but clearly, there is a single concept behind these words
- So we can think of this sequence of words as of a larger unit of language
- Languages have a compositional structure
- Languages are recursive, and this is particularly interesting to linguists (Chomsky)
- Recursion here means structural recursion
- Roughly, this means when parsing a sentence, you encounter a sub-sentence that is a sentence in itself and can contain sub-sentence that is a sentence in itself...
- Example: "[The man from [the company that you spoke with about [the project] yesterday]]"
- Can we find these larger units?
- Can we know when larger units are similar in meaning? E.g. "person on a snowboard" and "snowboarder"
- Words like "they", "there" allow us to refer to things mentioned before, can we make that link?
- You want to start with words: "the", "country", "of", "my", "birth" and be able to combine those in a single meaning of "the country of my birth"
- We already have vectors for words, but we need a way to compose things: some kind of composition function
- We need some kind of parser that can produce a tree structure, this is actually quite tricky
- Context-free grammar parsers tend to do very poorly, as they don't handle syntactic ambiguities well
- There is a whole line of work in this direction

### Implementation

- Take 2 word vectors, feed them as an input of a NN, should produce a combined meaning vector + plausibility score for this word combination
- Run greedy parser that picks the combination with the highest plausibility score
- As an improvement, instead of being greedy, you can use beam search
- We want to maximize the score for the tree: the sum of the scores at each node
- This requires backpropagation through the whole tree structure, splitting derivatives at each node
- When combining 2 children nodes, you can use different composition functions, depending on the nodes you combine
- For example, if you know you are combining a noun with the preposition, you use a composition function that knows how to combine those


## Coreference Resolution

- In simple terms: identify all noun phrases and connect them
- Example: correctly connect "Obama" to "his"/"him" or "the former president"
- This is an extremely complicated task
- There is a lot of nesting, "[CFO of [Prime Corp]]"
- Coreference is required for the full text understanding, but helps in many other tasks, like machine translation
- For example, in Turkish "he" and "she" is the same word, so you need to do coreference to translate sentences correctly
- Coreference is needed for information extraction
- Coreference, in some way, is similar to clustering, you are basically lumping several different parts of text together, and saying "this is the same thing"
- So people use the same metrics to evaluate the results of coreference as they do for clustering algorithms
- You can easily calculate precision and recall
- One of the popular metrics used to evaluate coreference is B-CUBED, which is based on precision and recall

### Anaphoric relationships

- To really really understand text, you would also need to understand anophoric relationships
- Anophoric relationships happen when to understand what you are talking about, you need the previous context
- Example: "We went to see [a concert] last night. [The tickets] were really expensive". "The tickets" refer to the tickets for the "concert" we went to
- This is anaphoric, but not coreferencial relationship
- Turns out, you really need a knowledge about the world to understand text
- Example: "[The city council] refused [the women] a permit because they feared violence" vs "[The city council] refused [the women] a permit because they advocated violence", "they" refers to "The city council" in one case and "the women" in another
- In 1976, Hobbs proposed Hobbs Algorithm, which is based on heuristics, which is often used until today to extract features
- The Hobbs Algorithm walks dependency tree to find the relationships
- There is almost no deep learning applied to solving this problem, the few existing ones use hand-crafted features


## Question Answering

- We could reframe every problem in NLP as a question answering
- Example 1: "Sandra went to the garden. She took the milk there. Where is the milk? Garden"
- Example 2: "Jane has a baby in Dresden. What are named entities? Jane - person, Dresden - location"
- Example 3: "I think this model is incredible. In French? Je pense que ce modèle est incroyable"
- Can we build a single joint model that can do all of these things?
- Dynamic Memory Network tries to do it
- Unfortunately, in many cases re-using layers for unrelated tasks actually hurts the performance, and dynamic memory networks do not solve it

### Dynamic Memory Network

- Start with GloVe vector representations (Semantic memory module)
- Run the input text through the standard GRU RNN and compute the internal hidden state for every sentence (Input module)
- Run the question through the GRU RNN and compute the question final state, the question vector (you may want to re-use the weights from RNN that runs through the whole text) (Question module)
- Use the question vector as an input into an attention function to find all the sentences of the input text that are relevant for answering the question
- Use those selected inner states an input to yet another GRU RNN (episodic memory module), weighted by the attention score
- Episodic memory module goes through those relevant sentence vectors and calculates the final vector for all the relevant sentences (as a last hidden state)
- This vector and the question vector are passed together to the softmax layer (answer module) that picks the answer word
- You may want to do multiple passes and calculate several states for the episodic memory module. With each pass you may discover that you need to pay attention to different sentences
- The power is, you can train this model end-to-end
- The model struggles with giving answers that it has never seen at training stage, but there are potential improvements
