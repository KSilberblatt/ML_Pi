# **Goals/ToDo**
*Update and change goals as they are completed and to keep track of what to learn*

* Learn about machine learning practices
  * Use/Learn Tensor Flow
    * Use/Learn Keras
      * regulizers?
      * dropout?
    * Optimizers?
    * Loss? (accuracy vs loss)
    * Metrics?


  * Understand how the ML process works
    * Neural networks -*see references*-
      * embedding layer
      * global average cooling layer
      * overfit and underfit
    * beginner alternatives to neural networks
      * k-nearest neighbors -*see references*-
      * decision trees
    * epochs?

* Learn how to program a Raspberry Pi 3 b+

* Decide how to handle multiple dice - options:
  * Image parsing - recognize objects that are dice
    * crop the image onto each die


* Make initial plan
* Structure markdown
* Structure repository
* Make MVP
* Make additional features section
* Play with Raspberry Pi
* Make rough UX
* Transition this ReadME out of note and into about project



### *Finished:*
> * buy materials
> * start github project
> * initialize notes structure

# Plan for project
## Create a dice tray that can sum the dice rolled and display the results
### Approach
#### Have a mounted camera(1) constantly analyze a designated field of view(2). when the camera notices a change in the FoV, snap a photo for analysis. This photo is sent to a computer(3). The computer will process the data and represent the result onto a display(4). Incorporate a GUI(5) for useful interaction with the hardware. Have a battery(6) so that the computer can work plugged in or unplugged.
1. ArduCam - 5 Megapixels, 1080p
  * maybe instead of constantly having the camera active, taking a photo every second would be more battery efficient
2. Red velvet stickers
3. Raspberry Pi 3 B+ (should have more than enough power for the features I want)
4. 7" touch screen display
5. --**Currently unknown**--
6. 4000mAh battery already regulates voltage to not fry the components


#### Analysis of photo: Convert image to grayscale (1). Use algorithm to detect the individual die in FoV (2). Convert image to an array of smaller images to analyze the individual die (3). Run image recognition to classify die (4). Output die values and die type into an array.
1. Methods base on luminescence of the photo. Might need some googling to discover the best formula for this use.
2. Probably not a ML problem. Just find clusters of changed pixel values and trim around them.
3. Saving new photos might be inefficient maybe I could just analyze the photo but only through certain pixel positions.
4. Use Tensor Flow for classification
  * 3 and 4 are probably going to be the rate limiting steps how can I speed this up?

### UI
#### All inclusive "single-page app" launched from desktop, could download chrome on the Pi and have the shortcut run a local server and run a React App
  * See MVP for features --*ToDo: Create MVPs*--
  * Might have to make custom styling since the app will probably be used while not connected to the internet. *Make this decision later*

# Notes

## **ML**
* High level what is a Neural network? -*see references*-
  * What is an embedding layer?
  * What is a dense layer?
  * What is a global average cooling layer?
  * How to handle overfit and underfit
    * regulizers?
    * dropout?
  * What is an Epoch?
* High level **TensorFlow**
  * Fun facts:
    * For image classification instead of starting from scratch, TensorFlow retrains *Inception* with your dataset. *Inception* took 2 weeks on a high powered desktop to train. Might be worth looking into to see how it was trained.
* High level **k-nearest neighbors**
    * A ML technique where the user collects training data in the form of a table that contains many samples coupled with its features and its label. The features are used to plot the label onto a multi-dimensional graph. The "black box" algorithm then creates a "classifier" in the form a function *k()* which determines how many close labeled neighbors (and more) input has. *k()* is changed to determine the best weight and how many "close" (using Euclidean distance) neighbors to evaluate such that the algorithm will most accurately (according to the training data) lead to correct labeling.
* High level **decision trees**
  * A ML technique where the user collects training data in the form of a table that contains many samples coupled with its features and its label. The "black box" algorithm then creates a "classifier" in the form of a series/binary-tree of true or false questions that will most accurately (according to the training data) lead to correct labeling.
  * What makes a good feature:
    * a quality one or more of the labels may contain that sets it apart from other labels.
    * is informative, *independent*, and simple.
    * **in tensor flow features are automatically assigned** by the "black box" algorithm

### *Key Terms:*
* Classifier: a function that takes data as an input and outputs a label effectively creating the "rules" that categorize your labels through **supervised learning**

## **Keras**
* What is activation?
* What are regulizers?
* What are optimizers?
* What does Loss measure?
* What are Metrics?
* What is eager execution?
* What is embedding dimension?

## **Raspberry Pi**



## **References**
* [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) ~10 hours
* [What is going on in a neural network?](www.distill.pub)
* [k-nearest neighbors YT example](https://www.youtube.com/watch?v=AoeEHqVSNOw) ~9 min
* [Tensor flow probability - math heavy](https://www.tensorflow.org/probability)
* [Tensor flow YT doc on github](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/basic_classification.ipynb)
* [Stanford Tensor flow course](http://web.stanford.edu/class/cs20si/)
* [Stanford Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/syllabus.html)
* [YouTube - Machine Learning Recipes #1](https://www.youtube.com/watch?v=cKxRvEZd3Mw) (1)
* [Book/PDF Deep Learning with Python](http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf)
* [Book/PDF Hands on Machine Learning with Scikit-Learn & Tensor Flow](http://index-of.es/Varios-2/Hands%20on%20Machine%20Learning%20with%20Scikit%20Learn%20and%20Tensorflow.pdf)
* [Book/PDF Deep Learning](https://www.deeplearningbook.org/front_matter.pdf)

## **Important random notes**
*delete when memorized or when sorted in a better place*
* Randomize training data and order of input
* When to stop training data? -When loss is no longer decreasing *early stopping method in Keras*

## Fun Examples
* [QuickDraw](https://quickdraw.withgoogle.com/)
* [QuickDraw Auto-complete](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html)
* [PacMan ML webcam controls](https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html)
* [Tensor flow playground](playground.tensorflow.org)

## Tools
* [Visualize data with Facets](https://github.com/PAIR-code/facets)
