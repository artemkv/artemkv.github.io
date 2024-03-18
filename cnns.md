# CNNs
{:.no_toc}

* A markdown unordered list which will be replaced with the ToC, excluding the "Contents header" from above
{:toc}

## References

- [Stanford CS 230: Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)


## Motivation

- A fully connected feedforward neural networks can learn features as well as classify data
- However, connecting 1 neuron to 1 pixel and then having multiple layers of fully connected neurons is impractical, as it will require a huge number of neurons, which means long computational time and huge risk of overfitting
- Plus, NN input layer is a vector, so an image would have to be flattened, which would result in losing spacial relationship between pixels
- To combat this, we try to split the problem into 2 separate problems and use 2 different network architectures for solving each task in an optimal way: convolutional layer for detecting features and the fully connected layer for actual classification
- **Convolution** is a process of applying filter (very small matrix with a certain pattern) to the original image (sliding the filter across the image) with the goal of detecting features


## Convolutional network general architecture

- The input layer will hold the raw pixel values of the image, just as before
- The convolutional layer will detect features. Each neuron will only connect to a limited number of input layer neurons
- The convolutional layer is essentially build by applying a filter on an original image
- Optional pooling layer can be used to downsample the image by combining information from several pixels into one
- The fully connected layer will do classification, just as before


## Convolutional network details

- When described above, layers were 1-directional vectors. This time every layer will have have neurons logically arranged in 3 dimensions: width, height, depth
- The word _depth_ previously used to refer to number of layers in this case refers to the 3rd dimension of the same layer
- In other words, neurons in the 3-dimensional layer are not interconnected, but only forward-connected to the neurons in the next 3-dimensional layer
- For the input layer, in case of an image, first 2 dimensions are the width and the height of the image, and the 3rd dimension is RGB channels
- For the convolutional layer, first 2 dimensions match the width and the height of the picture (almost true, depends on stride and zero-padding, see below)
- Despite the fact that convolutional layer has the same width and the height as the input layer, number of connections is small, since 1 neuron in the convolutional layer is only connected to a limited number of neurons of the input layer
- This limited number of connections, when 1 neuron is limited to a very narrow **receptive field** gives the effect of applying a filter on the original image: we get 1 output value for the whole area of the receptive field, and we have the largest output when pixels in the receptive field match exactly the weighs of the connections
- So technically speaking, the filter is the 3d matrix of weights for the connections within the receptive field
- A typical filter on a first layer of a convolutional network has size 5*5*3 (3 for 3 RGB dimensions). 3*3*3 and 5*5*3 sizes are the most used. Less commonly, larger sizes are used (7*7*3) but only in the first convolutional layer. So that means 5*5*3 = 75 incoming weights into a convolutional layer neuron
- Unlike the height and the width, the depth of the filter always matches the depth of the input layer
- The 3rd dimension of the convolutional layer holds results of applying different filters on the same area of the original input (the same receptive field), one dimension per filter (for example, 12 filters)
- Filters are not hard-coded presets, they are initialized randomly and adjusted using backpropagation algorithm, resulting in filters adapting to the images. So we end up with optimal filters that emerge from training
- As for the different areas of the image, you have a choice of using the same filter or different filters
- Using the same filter allows optimizing the calculations, since you need to store less data. Also, you are able to recognize the same pattern regardless of their position on the image
- Using different filters for different areas can be useful when different areas of the image have different objects
- The last layer is flattened and passed to the fully-connected layer


## Second layer of convolution

- Filters always extend the full depth of the input volume. This means the second convolution layer will treat the activation maps created by the previous convolution layer as channels
- So every filter in the second convolution will work on all activation maps (or feature maps) produced by the first convolution
- If image is RGB, you have 3 channels initially. If your first 3x3 convolution layer has 8 filters, you are going to use 3x3x3 weight tensors to produce 8 activation maps
- If your second 5x5 convolution layer has 16 filters, you are going to use 8x5x5 weight tensors to produce 16 activation maps


## Convolutional network implementation details

- If we use the same filter all across the image, you can think of a single neuron connected to 5*5 area of the input, sliding over the whole image using the same weights to calculate the 2-dimensional activation map. This neuron is a single filter. Then we slide another neuron with different weights across the image, producing the second dimension of the results
- When building such a model, we use depth, stride and zero-padding as hyperparameters
- Depth in this case corresponds to the number of different filters we would like to use
- Stride denotes how many pixels we move at the time when sliding across the picture. Having the stride of 2 reduces the width and the height of the output by 2
- Zero-padding means pad the input volume with zeros around the border. It is used to preserve exactly the same width and the height as the input
- In this implementation, we update the weights based on the gradient computed across the whole picture surface, once per slice
- In practice, instead of defining a filter as a 3d tensor and sliding it across an image, we pre-calculate the matrix where the same filter appears in every row at offsets that would match the correct areas of an image flattened into a vector. So in the end we are performing matrix multiplications


## Inception Module (GoogLeNet)

- In the past, most popular CNNs just stacked convolution layers deeper and deeper, hoping to get better performance
- But very deep networks are prone to overfitting (or at least that was a hypothesis at the time)
- Plus, naively stacking large convolution operations is computationally expensive
- Instead, you can have filters with different sizes operate on the same level
- You can even have 1x1 convolutions which are very cheap
- So you run the convolution with 3 different sizes of filters (1x1, 3x3, 5x5) and then concatenate them together (depth-wise)
- This forms an "inception module"
- Popular neural network architecture GoogLeNet has 9 inception modules stacked linearly


## Residual CNNs (ResNets)

- ResNets start from the fact that deeper CNNs don't seem to be overfitting in practice, they just "struggle to learn"
- Instead of forcing every layer to learn `H(x)` directly let's make it to learn `F(x) = H(x) - x`
- The intuition: instead of learning a completely new output, it may be easier to learn how to transform an input to get output
- You achieve that by simply adding `x` back to the conv output before ReLU
- ResNet is a current best default


## Pooling

- Pooling is a form of non-linear down-sampling
- The pooling layer serves to progressively reduce the spatial size of the representation, to reduce the number of parameters, memory footprint and amount of computation in the network, and hence to also control overfitting
- It is common to periodically insert a pooling layer between successive convolutional layers in a CNN architecture
- Basically, you combine 4 adjacent pixels into just 1, by using the maximum or average value
- The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2
- Very large input volumes may warrant 4x4 pooling in the lower layers
- Max pooling performs better than average pooling, so it is most commonly used in practice
- The intuitive reason is: max pooling allows you, for a given filter, to attribute an extra importance to certain kind of feature (you see a strong signal, and you train your filter to drive this signal even stronger for that particular feature)


## Embedding and autoencoders

- Convolutional networks reduce the number of parameters to combat overfitting, but still require large amounts of labeled training data that is scarce and expensive to generate
- Embedding works when labeled data is scarce but wild, unlabeled data is plentiful
- Embedding is learning low-dimensional representations in an unsupervised fashion. Embeddings can be later used for the supervised training on less data (HOW EXACTLY?)
- Autoencoder is an architecture with encoder and decoder. Encoder is used to produce an embedding (or code), decoder is trying to invert the computation and reconstruct the original input
- When running on MNIST dataset, after several epochs (like 200) we can obtain pretty smooth reconstructions of the digits


## Data preparation and initialization

- Important: zero-center the data. You can calculate mean for the whole image (i.e. all channels) (AlexNet), or calculate and subtract mean for each channel individually (VGGNet)
- Normalize on the scale between -1 and 1. Not strictly necessary for images, since the relative scales of pixels are already approximately equal (and in range from 0 to 255


## Semantic segmentation

- For every pixel on the image, return a category
- One approach is to apply image classification task on a small window of an image to classify the central pixel, and slide that window to classify every pixel. This would be very computationally expensive, so no one does this in practice
- Another approach is to stack a huge number of convolutional layers on top of each other, preserving the original image dimensions, and then in the end a single convolutional layer of shape (`C`, `H`, `W`) where `C` is a number of categories. Every pixel would have its own cross-entropy loss. This can work, but this would still be extremely computationally expensive. So this is not done in practice neither
- What is done in practice looks like an autoencoder architecture. The image is downsampled, classified using convolutional layers and then upsampled back. Downsampling/upsampling can be done using pooling/unpooling. But you can also upsample using **transpose convolution**
- Transpose convolution is kind of reverse of a normal convolution, so sometimes it is called "deconvolution" or "upconvolution". This leads to "learnable" upsampling
- Creating the training data for semantic segmentation is super expensive, since you need to label every pixel
- Does not differentiate instances, only care about pixels. So multiple objects of the same type can get "fused together"


## Classification + localization

- In addition to classifying an image as "cat", draw a bounding box around the cat
- The approach is to use CNN that in addition to classification, returns box coordinates. In this case, we would need to 2 losses (multitask loss)


## Object detection

- Draw a bounding box around every object that represents any category you care about
- You can have varying number of objects on the same picture
- One approach is using a sliding window, but it's super difficult to come up with the right sliding window sizes and positions. So in practice, people don't do this
- What is done is pre-computing regions that are likely to contain objects ("region proposals") to get about 1000-2000 region proposals. Then you run CNN to predict categories and adjustments for the regions
- You can do region proposals on top of CNN layers to re-use some computations
 
