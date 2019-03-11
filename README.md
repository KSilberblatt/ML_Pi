# **Goals/ToDo**
*Update and change goals as they are completed and to keep track of what to learn*

* Look into [OpenCV](https://opencv.org/)

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
  * Hyperparameter tuning
    * Is There a Standard Heuristic for Model Tuning? :This is a commonly asked question. The short answer is that the effects of different hyperparameters are data dependent. So there are no hard-and-fast rules; you'll need to test on your data. That said, here are a few rules of thumb that may help guide you:
      * Training error should steadily decrease, steeply at first, and should eventually plateau as training converges.
      * If the training has not converged, try running it for longer.
      * If the training error decreases too slowly, increasing the learning rate may help it decrease faster.
        * But sometimes the exact opposite may happen if the learning rate is too high.
      * If the training error varies wildly, try decreasing the learning rate.
        * Lower learning rate plus larger number of steps or larger batch size is often a good combination.
      * Very small batch sizes can also cause instability. First try larger values like 100 or 1000, and decrease until you see degradation.
    * Never go strictly by these rules of thumb, because the effects are data dependent. Always experiment and verify.
    * Common hyperparameters: steps and batch size
      * <code>*total number of trained examples* = *batch size* \*
      *steps*</code>
      * steps: the total number of training iterations. One step calculates the loss from one batch and uses that value to modify the model's weights once
      * batch size: the number of examples (chosen at random) for a single step
    * [More on Hyperparameters, Cleaning Data, and plotting](https://colab.research.google.com/notebooks/mlcc/synthetic_features_and_outliers.ipynb?utm_source=mlcc&utm_campaign=colab-external&utm_medium=referral&utm_content=syntheticfeatures-colab&hl=en)
  * Convenience variable: a variable to help the programmer visualize data but doesn't affect the outcome of what the machine learns
    * Periods: controls the granularity of reporting. For example, if periods is set to 7 and steps is set to 70, then the exercise will output the loss value every 10 steps (or 7 times). Unlike hyperparameters, we don't expect you to modify the value of periods. Note that modifying periods does not alter what your model learns.
      <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;number\,of\,training\,examples\,in\,each\,period&space;=&space;\frac{batch\,size&space;*&space;steps}&space;{periods}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;number\,of\,training\,examples\,in\,each\,period&space;=&space;\frac{batch\,size&space;*&space;steps}&space;{periods}" title="number\,of\,training\,examples\,in\,each\,period = \frac{batch\,size * steps} {periods}" /></a>
  * **Generalization**... the following three basic assumptions guide generalization:
    * We draw examples independently and identically **(i.i.d)** at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: **i.i.d.** is a way of referring to the randomness of variables.)
    * The distribution is stationary; that is the distribution doesn't change within the data set.
    * We draw examples from partitions from the same distribution.

    In practice, we sometimes violate these assumptions. For example:
    * Consider a model that chooses ads to display. The **i.i.d.** assumption would be violated if the model bases its choice of ads, in part, on what ads the user has previously seen.
    * Consider a data set that contains retail sales information for a year. User's purchases change seasonally, which would violate stationarity.
  * What is an embedding layer?
  * What is a dense layer?
  * What is a global average cooling layer?
  * How to handle overfit and underfit
    * Ockham's razor in machine learning terms: The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample. Helps to avoid overfitting
      * modern fields following this principle are **statistical learning theory** and **computational learning theory**. These fields have developed generalization bounds--a statistical description of a model's ability to generalize to new data based on factors such as:
        * the complexity of the model
        * the model's performance on training data
    * regulizers?
    * dropout?
  * What is an Epoch?
* High level **TensorFlow**: a computational framework for building machine learning models. TensorFlow provides a variety of different toolkits that allow you to construct models at your preferred level of abstraction. You can use lower-level APIs to build models by defining a series of mathematical operations. Alternatively, you can use higher-level APIs (like tf.estimator) to specify predefined architectures, such as linear regressors or neural networks.
  * Data structures:
    * DataFrame: relational data table with rows and named columns. Can be populated by **Series** objects
      * Often DataFrames are populated by *.csv* files
    * Series: a single column of data, used to populate **DataFrames** when coupled with names
      * if the **Series** objects' length don't match during mapping NaN values are assigned in place.
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
* ML Workflow notes: general facts/guidelines to remember during ML workflow
  * Splitting data into training data and test data:
    * The larger the training set the better model the machine can learn
    * The larger the testing set, the tighter the confidence interval we can have in our algorithm
    * when splitting make sure the examples are split at random
    * never train on test data - **sometimes happens on accident**
    * using the same test set too many times can also lead false confidence since hyperparameters will be tuned to that test set.
      * validation set: similar to a test set, however, by creating a validation set and tuning your hyperparameters to the validation set, you keep the test set clean from over fitting
  * Features:
    * when using categorical data as a feature, **one-hot encoding** helps with many categories that may belong to that feature
      * one-hot encoding can be represented by a vector where '*1*' represents the category the data entry belongs to, [0, 0, ..., 0, 1, 0, ..., 0]
    * Good features:
      * Occurs at least a handful of times or more
      * Have a clear and obvious meaning
      * Shouldn't have "magic values". i.e. listed houses and past sales where "-1" might indicate it hasn't been listed or sold yet.
      * Shouldn't change overtime
      * Shouldn't have crazy outliers
    * The **Binning Trick**: for features without linear correlation over the entire feature, but strong linear correlation within subsets of the feature.
      * bucket the the subset features into boolean features (**one-hot encoding**)
  * Know your data: Visualize, debug, monitor -
    * Visualize: plot histograms, rank most to least common
    * Debug: Duplicate examples? missing values? outliers? data agrees with dashboards? Training and Validation data similar?
    * Monitor: feature quantiles, number of examples over time?

### *Key Terms:*
* Generalization: your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.
* Stationarity: A property of data in a dataset, in which the data distribution stays constant across one or more dimensions. Most commonly, that dimension is time, meaning that data exhibiting stationarity doesn't change over time. For example, data that exhibits stationarity doesn't change from September to December.
* Classifier: a function that takes data as an input and outputs a label effectively creating the "rules" that categorize your labels through **supervised learning**
* Labels: the thing we're predicting
* Model: A model defines the relationship between features and label. For example, a spam detection model might associate certain features strongly with "spam"
  * A **regression model** predicts continuous values
  * A **classification model** predicts discrete values
* Inference: in machine learning, often refers to the process of making predictions by applying the trained model to unlabeled examples. In statistics, inference refers to the process of fitting the parameters of a distribution conditioned on some observed data.
* Linear regression: a method for finding the straight line or hyperplane that best fits a set of points **line of best fit**
  * *y' =  b + w<sub>1</sub>x<sub>1</sub>*
    * y': our linear function
    * b: our y-intercept also our **bias**
      * bias: An intercept or offset from an origin. Bias (also known as the bias term) is referred to as b or w0 in machine learning models
    * w<sub>1</sub>: weight of feature 1 (slope of our line)
    * x<sub>1</sub>: feature 1 (a known input)
    * If there were more features we would add w's and x's up to *n* features.
      *y' =  b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub>*
* Loss (*L<sub>1</sub> loss*): Loss function based on the absolute value of the difference between the values that a model is predicting and the actual values of the labels. L<sub>1</sub> loss is less sensitive to outliers than L<sub>2</sub> loss.
* Squared Loss (*L<sub>2</sub> loss*): The loss function used in linear regression. This function calculates the squares of the difference between a model's predicted value for a labeled example and the actual value of the label. Due to squaring, this loss function amplifies the influence of bad predictions. That is, squared loss reacts more strongly to outliers than L1 loss.
* Mean square error (MSE) the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:
  * Formula: <a href="https://www.codecogs.com/eqnedit.php?latex=MSE&space;=&space;\frac{1}{N}&space;\sum_{(x,y)\in&space;D}&space;(y&space;-&space;prediction(x))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{N}&space;\sum_{(x,y)\in&space;D}&space;(y&space;-&space;prediction(x))^2" title="MSE = \frac{1}{N} \sum_{(x,y)\in D} (y - prediction(x))^2" /></a>
    * (*x,y*): an example in which
      * *x*: the set of features that the model uses to make predictions.
      * *y*: the example's label.
    * *prediction(x)*: function of the weights and bias in combination with the set of features *x*.
    * *D* is a data set containing many labeled examples, which are *(x,y)* pairs.
    * *N* is the number of examples in *D*.
* Empirical Risk Minimization (**ERM**): Choosing the function that minimizes loss on the training set. **Contrast** with structural risk minimization
* Structural Risk Minimization (**SRM**):
  * An algorithm that balances two goals:
    * The desire to build the most predictive model (for example, lowest loss).
    * The desire to keep the model as simple as possible (for example, strong regularization).
  * For example, a function that minimizes loss+regularization on the training set is a structural risk minimization algorithm.
  * For more information, see http://www.svms.org/srm/.
  * **Contrast** with empirical risk minimization.
* Convergence: Informally, often refers to a state reached during training in which training loss and validation loss change very little or not at all with each iteration after a certain number of iterations. In other words, a model reaches convergence when additional training on the current data will not improve the model. In deep learning, loss values sometimes stay constant or nearly so for many iterations before finally descending, temporarily producing a false sense of convergence.
  * See also [early stopping](https://developers.google.com/machine-learning/glossary/#early_stopping)
  * See also [Boyd and Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)
* Gradient descent: A technique to minimize loss by computing the gradients of loss with respect to the model's parameters, conditioned on training data. Informally, gradient descent iteratively adjusts parameters, gradually finding the best combination of weights and bias to minimize loss.
* Step: A forward and backward evaluation of one batch.
* Batch: The set of examples used in one iteration (that is, one gradient update) of model training.
* Learning rate (Also **step-size**): A scalar used to train a model via gradient descent. During each iteration, the gradient descent algorithm multiplies the learning rate by the **gradient**. The resulting product is called the gradient step.
  * Learning rate is a key **hyperparameter**.
  * **Gradient** or **gradient vector**: The vector of partial derivatives with respect to all of the independent variables. In machine learning, the gradient is the vector of partial derivatives of the model function. The gradient points in the direction of steepest ascent.
* Hyperparameter: The "knobs" that you (the programmer) tweak during successive runs of training a model.
* Stochastic gradient descent (Vanilla SGD): A gradient descent algorithm in which the **batch size** is one. In other words, SGD relies on a single example chosen uniformly at random from a dataset to calculate an estimate of the gradient at each step.
* Batch size: The number of examples in a batch. For example, the batch size of SGD is 1, while the batch size of a **mini-batch** is usually between 10 and 1000. Batch size is usually fixed during training and inference; however, TensorFlow does permit dynamic batch sizes.
* Mini-batch: A small, randomly selected subset of the entire batch of examples run together in a single iteration of training or inference. The batch size of a mini-batch is usually between 10 and 1,000. It is much more efficient to calculate the loss on a mini-batch than on the full training data.
  * Mini-batch stochastic gradient descent (Mini-batch SGD): A gradient descent algorithm that uses mini-batches. In other words, mini-batch SGD estimates the gradient based on a small subset of the training data. Vanilla SGD uses a mini-batch of size 1.
* Estimator: an instance of the <code>tf.Estimator</code> class, which encapsulates logic that builds a TensorFlow graph and runs a TensorFlow **session**. You may create your own custom Estimators or instantiate premade Estimators created by others.
  * Session (<code>tf.session</code>): An object that encapsulates the state of the TensorFlow runtime and runs all or part of a graph. When using the low-level TensorFlow APIs, you instantiate and manage one or more <code>tf.session</code> objects directly. When using the Estimators API, Estimators instantiate session objects for you.
* Tensor: The primary data structure in TensorFlow programs. Tensors are N-dimensional (where N could be very large) data structures, most commonly scalars, vectors, or matrices. The elements of a Tensor can hold integer, floating-point, or string values.
* Graph: In TensorFlow, a computation specification. Nodes in the graph represent operations. Edges are directed and represent passing the result of an **operation** (a Tensor) as an operand to another operation. Use **TensorBoard** to visualize a graph.
  * Operation (op): A node in the TensorFlow graph. In TensorFlow, any procedure that creates, manipulates, or destroys a Tensor is an operation. For example, a matrix multiply is an operation that takes two Tensors as input and generates one Tensor as output.
* Gradient clipping: Capping gradient values before applying them. Gradient clipping helps ensure numerical stability and prevents exploding gradients.



## **Keras**
* What is activation?
* What are regulizers?
* What are optimizers?
* What does Loss measure?
* What are Metrics?
* What is eager execution?
* What is embedding dimension?

## **Raspberry Pi**

## **Training Data**
### Choose method:
#### Two

## **References**
* [OpenCV solution for clusterd die](https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html?fbclid=IwAR3rmo0HVsDTJky-bHeUpX8xoRES1iZjNli6rnDSJrTBsHBz7wjzgwgvHwY)
* [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) ~10 hours [Bookmark](https://developers.google.com/machine-learning/crash-course/representation/cleaning-data)
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
* [Gradient Exploding](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf)

## **Important random notes**
*delete when memorized or when sorted in a better place*
* Randomize training data and order of input
* When to stop training data? -When loss is no longer decreasing *early stopping method in Keras*
**Potential accidental issues I introduce**
* not all n-sided die are rolled as frequently as one another
* when creating test/training data, I obviously want multiple photos of each side of every die I have, but could having **5 times** as many test/training photos of a 20-sided die vs a 4-sided die cause problems?
  * hint: probably

## Fun Examples
* [QuickDraw](https://quickdraw.withgoogle.com/)
* [QuickDraw Auto-complete](https://magenta.tensorflow.org/assets/sketch_rnn_demo/index.html)
* [PacMan ML webcam controls](https://storage.googleapis.com/tfjs-examples/webcam-transfer-learning/dist/index.html)
* [Tensor flow playground](playground.tensorflow.org)

## Tools
* [Visualize data with Facets](https://github.com/PAIR-code/facets)
* [Write Equations in Markdown](https://www.codecogs.com/latex/eqneditor.php)

## Notable sources I gathered and **copied** information (not bothering with MLA since these docs are not intended for distribution):
* https://developers.google.com/machine-learning/glossary
* https://developers.google.com/machine-learning/crash-course
* https://www.youtube.com/watch?v=tYYVSEHq-io&t=5496s
* https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal
