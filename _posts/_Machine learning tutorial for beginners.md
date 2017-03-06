
Machine learning tutorial for beginners
===================


Questions like "What is the stock price of next day?" or "What is the weather tomorrow?" is very difficult to give answer because there is no perfect algorithm can predict the future with numerical calculation.  **Machine learning**[^machinelearning] was born for this, which is subfield of computer science that gives computers the ability to learn without being explicitly programmed. There are many articles and paper discuss about what is machine learning and how to conduct experiments on machine learning,  this  will not explain the definition and concept that can be searched easily on the internet. I will go through the steps that people normally confused and get stuck in.  

> **Machine Learning solves the problem that cannot be solved by numerical means alone. **

There are two major types of machine learning can be used very often among research and industry:

 - **Supervised machine learning:** The program is “trained” on a pre-defined set of “training examples”, which then facilitate its ability to reach an accurate conclusion when given new data.
 - **Unsupervised machine learning:** The program is given a bunch of data and must find patterns and relationships therein.

--------------

Supervised Machine Learning
-----------------------------------------
In supervised learning, the ultimate goal is the find out the perfect function $h(x)$ (called hypothesis in machine learning) can calculate the result based on given data. The "learning" is normally a sophisticated mathematical calculation process for find the best parameters of the hypothesis function. Supervised learning also can divide into classification and regression process. Classification means samples belong to two or more classes and we want to learn from already labeled data how to predict the class of unlabeled data. An example of classification problem would be the handwritten digit recognition example, in which the aim is to assign each input vector to one of a finite number of discrete categories.  Regression means if the desired output consists of one or more continuous variables, then the task is called regression. An example of a regression problem would be the prediction of the length of a salmon as a function of its age and weight.





Simple Machine Learning Project in Python Step-By-Step
------------------------------------------
The general steps of a machine learning program is defined as:

1. Environment Installation.
2. Define Problem
2. Prepare Data.
3. Implement Algorithms.
4. Evaluate Results.
_______________________________________
**Environment Installation**
_______________________________________
It is highly recommend that using the Linux/UNIX for the machine learning environment set up,  I will use **Ubuntu**[^ubuntu] here.  **Scikit-learn**[^scikitlearn] is used as the library of machine learning algorithm provider. 

Install the dependencies on Ubuntu

```sh
$ sudo apt-get install python
$ sudo python -m pip install --upgrade pip
$ sudo pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
$ sudo pip install -U scikit-learn
```
For Mac OS user, Homebrew is a very efficient package manager, the install are nearly the same as Ubuntu:

```sh
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ sudo brew install python
$ sudo python -m pip install --upgrade pip
$ sudo pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
$ sudo pip install -U scikit-learn
```
_______________________________________
**Define Problem**
_______________________________________
In general, a learning problem considers a set of n samples of data and then tries to predict properties of unknown data. If each sample is more than a single number and, for instance, a multi-dimensional entry (aka multivariate data), it is said to have several attributes or features.
Machine learning is about learning some properties of a data set and applying them to new data. This is why a common practice in machine learning to evaluate an algorithm is to split the data at hand into two sets, one that we call the training set on which we learn data properties and one that we call the testing set on which we test these properties.
_______________________________________

**Prepare Data**
_______________________________________
The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.Two of the three species were collected in the Gaspé Peninsula "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus".[^iris]

Firstly, we start a python interpreter to load the iris dataset.
```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> digits = datasets.load_digits()
```
_______________________________________
**Implement Algorithm**
_______________________________________
```python
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
```
_______________________________________
**Evaluate Results**
_______________________________________
```python
clf.predict(digits.data[-1:])
array([8])
```
The related image is:
![alt text](http://scikit-learn.org/stable/_images/sphx_glr_plot_digits_last_image_001.png "iris image")

_______________________________________

Great Machine Learning Libraries
--------------------------------------

Good machine learning libraries work reliably, quickly, and at-scale. They can help engineers and scientists build, implement, and maintain production machine learning systems efficiently and easily. As a scientist, If we want to been able to see further, it was only because we stood on the shoulders of giants. Let's build great machine learning algorithms base on the existing excellent libraries.
There are also some other machine learning libraries can be found:

 - [Weka](http://www.cs.waikato.ac.nz/ml/weka/): this is a Java based library with a graphical user interface that allows you to run experiments on small datasets. This is great if you restrict yourself to playing around to get a feel for what is possible with machine learning. 
 - [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [LibLinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/): these are C libraries for support vector machines; there are also bindings or implementations for many other languages. These are the libraries used for support vector machine learning in Scikit-learn.
 - [Caffe](http://caffe.berkeleyvision.org/): is a deep learning framework made with expression, speed, and modularity in mind. It is developed by the Berkeley Vision and Learning Center (BVLC) and by community contributors. Yangqing Jia created the project during his PhD at UC Berkeley. Caffe is released under the BSD 2-Clause license. 
 - [TensorFlow](https://www.tensorflow.org/): is an open source software library for numerical computation using data flow graphs. TensorFlow implements what are called data flow graphs, where batches of data (“tensors”) can be processed by a series of algorithms described by a graph.
 - [Theano](http://www.deeplearning.net/software/theano/): efficient computation of mathematical expressions using GPU. Excellent for deep learning.
 - [Deeplearning4j](https://deeplearning4j.org/) is the first commercial-grade, open-source, distributed deep-learning library written for Java and Scala. Integrated with Hadoop and Spark, DL4J is designed to be used in business environments on distributed GPUs and CPUs.


[^machinelearning]:[Machine Learning Wiki ](https://en.wikipedia.org/wiki/Machine_learning) is the wiki page of machine learning, where you can get a basic understanding of machine learning.

[^ubuntu]:[Ubuntu ](https://www.ubuntu.com/) is built on Debian's architecture and infrastructure, to provide Linux server, desktop, phone, tablet and TV operating systems.

[^scikitlearn]:[Scikit-learn](http://scikit-learn.org/stable/index.html) is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

[^iris]:[Iris](https://archive.ics.uci.edu/ml/datasets/Iris) is  is perhaps the best known database to be found in the pattern recognition literature. Fisher's paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 


  [1]: http://math.stackexchange.com/
  [2]: http://daringfireball.net/projects/markdown/syntax "Markdown"
  [3]: https://github.com/jmcmanus/pagedown-extra "Pagedown Extra"
  [4]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
  [5]: https://code.google.com/p/google-code-prettify/
  [6]: http://highlightjs.org/
  [7]: http://bramp.github.io/js-sequence-diagrams/
  [8]: http://adrai.github.io/flowchart.js/
