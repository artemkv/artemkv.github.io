# RNNs

## References

- [Stanford CS 230: Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)


## Intro

- Traditional ML algorithms (including feedforward neural networks) do not work well with sequences
- ML assume data is independently and identically distributed, all drawn from some distribution, so they lose the contextual information
- For example, if we want to translate the sentence, the word order matters ("dog bites man" vs "man bites dog")
- ML assume fixed, finite number of features, but the sequences can be of different lengths, or even infinite


## RNN

- To "remember" the previous item, we could connect the neurons in the hidden layer back to themselves (recurrent connection)
- This way the output from the hidden layer is feed back into the hidden layer
- That connection will have its own set of weights, commonly denoted as `Whh` (hidden to hidden)
- More exactly, every neuron in the hidden layer connects to every other neuron in the hidden layer, so having `M` neurons in the hidden layer, `Whh` is a matrix `M*M`
- The values from the input layer are added to the values from the recurrent connections before being passed through the activation function (usually tanh)

```
h(t) = f(Whh*h(t-1) + Wxh*x(t)+ b)
```

- The value of the hidden layer at any time `h(t)` depends on its value at the previous step `h(t-1)`
- But `h(t-1)` depends on `h(t-2)` and so on. So this relation does not hold the Markov's property (Markov's property requires the function depend only on the immediately preceding state and action, not on any earlier states and actions)
- Note that the same weights are re-used across all time steps
- `h(0)` can be set to 0, or can be a hyperparameter
- The number of recurrent layers can vary, you can have more than one
- RNNs can be used both for regression and classification
- At the output layer we could have label at every time step, or simply at the last step
- Predicting the label over entire sequence could be useful, for example, for predicting the gender from the voice
- Predicting the label at every step could be useful for controlling devices
- Predicting the next value could be useful for predicting next word in the sentence

### Backpropagation in RNN

- Backpropagation has to be done through time (or BPTT)
- The error will be the function of every previous value, and when taking the derivative using the chain rule you will end up doing lots of multiplications
- This results in the problem of vanishing/exploding gradients (try multiplying 0.1 by itself a thousand times)
- Gradient clipping is a technique that tackles exploding gradients. The idea of gradient clipping is very simple: if the gradient gets too large, we rescale it to keep it small
- Another solution is to truncate the sum after a certain amount of steps (basically you only look a given amount of steps back). This leads to an approximation of the true gradient (the model focuses primarily on short-term influence rather than long-term consequences)


## Modern RNNs

- In theory, classic RNNs can keep track of arbitrary long-term dependencies in the input sequences
- In practice, RNNs suffer from a vanishing gradient problem
- Some very early observation can be highly significant for predicting all future observations
- Some sequence elements can be irrelevant and need to be skipped
- There might be a logical break between parts of a sequence, which requires resetting the internal state representation

### Gated Recurrent Units (GRU)

- **Gated Recurrent Units (GRU)** is an RNN architecture that supports "gating" of the previous hidden state: GRUs can learn when a hidden state should be updated or reset
- GRU contains 2 internal gates: "R" for **reset gate** and "Z" for **update gate** (go figure)
- Both gates produce a value that is calculated very similarly to the normal output:

```
R(t) = g(Whr*h(t-1) + Wxr*x(t) + br)
Z(t) = g(Whz*h(t-1) + Wxz*x(t) + bz)
```

- So basically, input and hidden input from previous step are multiplied by their own set of weights, combined and passed through a sigmoid
- Reset gate allows controlling how much of the previous state should be remembered
- Before being added to the input, the value from the previous step `h(t-1)` is multiplied with the value of R gate elementwise to produce candidate hidden state
- When the value of the reset gate is close to 1, the unit functions as a conventional RNN, keeping the most of the hidden state from the previous step
- When the value of the reset gate is close to 0, the previous hidden state is effectively thrown away
- Update gate allows controlling how much of the new state is just a copy of the old state (simply put, should we ignore the current sequence element or not)
- Before being passed to the output, the candidate hidden state is multiplied with the inverse value of `Z` gate elementwise
- When the value of the update gate is close to 1, the candidate hidden state is ignored, and hidden state from the previous step becomes the new output
- When the value of the update gate is close to 0, the candidate hidden state becomes the new output

### Long Short Term Memory (LSTM)

- **Long Short Term Memory (LSTM)** is similar to GRU, except it is much older (1997) and more complex
- Instead of 2, LSTM contains 3 internal gates: "I" for input gate, "O" for output gate and "F" for forget gate
- The values they produce are calculated the exact same way as in GRU
- **The candidate memory cell** is calculated similar to how classic RNN output is calculated, combining the input and the hidden value at the previous step `h(t-1)` and using tanh as an activation function
- After that, the candidate memory cell and the memory cell from the previous step are multiplied with the values of I gate and F gate respectively, producing new **memory cell**
- Before being passed to the output, this memory cell is passed through tanh and then multiplied with the value of O gate, finally producing the new hidden layer output
- LSTM produces 2 outputs: new hidden layer output and the new memory cell
- LSTMs can cope with vanishing and exploding gradients
