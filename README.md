# cse446-homework-3-solved
**TO GET THIS SOLUTION VISIT:** [CSE446-Homework 3 Solved](https://mantutor.com/product/cse446-homework-3-solved/)


---

**For Custom/Order Solutions:** **Email:** mantutorcodes@gmail.com  

*We deliver quick, professional, and affordable assignment help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;71770&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE446-Homework 3 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Homework 3

CSE 446: Machine Learning

<ul>
<li></li>
</ul>
<h1>Conceptual Questions</h1>
<ol>
<li>The answers to these questions should be answerable without referring to external materials. Briefly justify your answers with a few words.</li>
<li><em>[2 points] </em>True or False: The maximum margin decision boundaries that support vector machines construct have the lowest generalization error among all linear classifiers.</li>
<li><em>[2 points] </em>Say you trained an SVM classifier with an RBF kernel (). It seems to underfit the training set: should you increase or decrease <em><sub>σ</sub></em>?</li>
<li><em>[2 points] </em>True or False: Training deep neural networks requires minimizing a non-convex loss function, and therefore gradient descent might not reach the globally-optimal solution.</li>
<li><em>[2 points] </em>True or False: It is a good practice to initialize all weights to zero when training a deep neural network.</li>
<li><em>[2 points] </em>True or False: We use non-linear activation functions in a neural network’s hidden layers so that the network learns non-linear decision boundaries.</li>
<li><em>[2 points] </em>True or False: Given a neural network, the time complexity of the backward pass step in the backpropagation algorithm can be prohibitively larger compared to the relatively low time complexity of the forward pass step.</li>
</ol>
<h1>Kernels</h1>
<ol start="2">
<li><em>[5 points] </em>Suppose that our inputs <em><sub>x </sub></em>are one-dimensional and that our feature map is infinite-dimensional: <em>φ</em><sub>(<em>x</em>) </sub>is a vector whose <em><sub>i</sub></em>th component is</li>
</ol>
for all nonnegative integers <em><sub>i</sub></em>. (Thus, <em><sub>φ </sub></em>is an infinite-dimensional vector.) Show that is a

kernel function for this feature map, i.e.,

<em>.</em>

Hint: Use the Taylor expansion of <em><sub>e</sub></em><em><sup>z</sup></em>. (This is the one dimensional version of the Gaussian (RBF) kernel). 3. This problem will get you familiar with kernel ridge regression using the polynomial and RBF kernels. First, let’s generate some data. Let <em>n </em>= 30 and <em>f</em><sub>∗</sub>(<em>x</em>) = 4sin(<em>πx</em>)cos(6<em>πx</em><sup>2</sup>). For <em>i </em>= 1<em>,…,n </em>let each <em>x<sub>i </sub></em>be drawn uniformly at random on <sub>[0<em>,</em>1] </sub>and where.

For any function <em><sub>f</sub></em>, the true error and the train error are respectively defined as

<em>.</em>

Using kernel ridge regression, construct a predictor

<em>n </em>= argmin||<em>Kα </em>− <em>y</em>||<sup>2 </sup>+ <em>λα<sup>T</sup>Kα ,&nbsp;&nbsp;&nbsp; f</em><sub>b</sub>(<em>x</em>) = <sup>X</sup><em>α</em><sub>b</sub><em>i</em><em>k</em>(<em>x<sub>i</sub>,x</em>)

<em><sub>α</sub></em>

<em>i</em>=1

where <em><sub>K</sub></em><em><sub>i,j </sub></em><sub>= <em>k</em>(<em>x</em></sub><em><sub>i</sub></em><em><sub>,x</sub></em><em><sub>j</sub></em><sub>) </sub>is a kernel evaluation and <em><sub>λ </sub></em>is the regularization constant. Include any code you use for your experiments in your submission.

<ol>
<li><em>[10 points] </em>Using leave-one-out cross validation, find a good <em><sub>λ </sub></em>and hyperparameter settings for the following kernels:
<ul>
<li><em>k<sub>poly</sub></em>(<em>x,z</em>) = (1 + <em>x<sup>T</sup>z</em>)<em><sup>d </sup></em>where <em>d </em>∈ N is a hyperparameter,</li>
<li><em>k<sub>rbf</sub></em>(<em>x,z</em>) = exp(−<em>γ</em>k<em>x </em>− <em>z</em>k<sup>2</sup>) where <em>γ &gt; </em>0 is a hyperparameter<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Report the values of <em><sub>d</sub></em>, <em><sub>γ</sub></em>, and the <em><sub>λ </sub></em>values for both kernels.</li>
</ul>
</li>
<li><em>[10 points] </em>Let and be the functions learned using the hyperparameters you found in part</li>
<li>For a single plot per function <sub>b </sub>, plot the original data, the true <em><sub>f</sub></em><sub>(<em>x</em>)</sub>, and (i.e., define a fine grid on <sub>[0<em>,</em>1] </sub>to plot the functions).</li>
</ol>
<h1>Neural Networks for MNIST</h1>
<ol start="4">
<li>In Homework 1, we used ridge regression for training a classifier for the MNIST data set. In Homework 2, we used logistic regression to distinguish between the digits 2 and 7. In this problem, we will use PyTorch to build a simple neural network classifier for MNIST to further improve our accuracy.</li>
</ol>
We will implement two different architectures: a shallow but wide network, and a narrow but deeper network. For both architectures, we use <em><sub>d </sub></em>to refer to the number of input features (in MNIST, <em><sub>d </sub></em><sub>= 28</sub><sup>2 </sup><sub>= 784</sub>), <em><sub>h</sub></em><em><sub>i </sub></em>to refer to the dimension of the <em><sub>i</sub></em>th hidden layer and <em><sub>k </sub></em>for the number of target classes (in MNIST, <em><sub>k </sub></em><sub>= 10</sub>). For the non-linear activation, use ReLU. Recall from lecture that

ReLU

<h2>Weight Initialization</h2>
Consider a weight matrix <em><sub>W </sub></em><sub>∈ </sub>R<em><sup>n</sup></em><sup>×<em>m </em></sup>and <em><sub>b </sub></em><sub>∈ </sub>R<em><sup>n</sup></em>. Note that here <em><sub>m </sub></em>refers to the input dimension and <em><sub>n </sub></em>to the output dimension of the transformation <em><sub>Wx </sub></em><sub>+ <em>b</em></sub>. Define. Initialize all your weight matrices and biases according to Unif<sub>(</sub>−<em><sub>α,α</sub></em><sub>)</sub>.

<h2>Training</h2>
For this assignment, use the Adam optimizer from torch.optim. Adam is a more advanced form of gradient descent that combines momentum and learning rate scaling. It often converges faster than regular gradient descent. You can use either Gradient Descent or any form of Stochastic Gradient Descent. Note that you are still using Adam, but might pass either the full data, a single datapoint or a batch of data to it. Use cross entropy for the loss function and ReLU for the non-linearity.

<h2>Implementing the Neural Networks</h2>
<ol>
<li><em>[10 points] </em>Let <em>W</em><sub>0 </sub>∈ R<em><sup>h</sup></em><sup>×<em>d</em></sup>, <em>b</em><sub>0 </sub>∈ R<em><sup>h</sup></em>, <em>W</em><sub>1 </sub>∈ R<em><sup>k</sup></em><sup>×<em>h</em></sup>, <em>b</em><sub>1 </sub>∈ R<em><sup>k </sup></em>and <em>σ</em>(<em>z</em>) : R → R some non-linear activation function. Given some <em><sub>x </sub></em><sub>∈ </sub>R<em><sup>d</sup></em>, the forward pass of the wide, shallow network can be formulated as:</li>
</ol>
F<sub>1</sub>(<em>x</em>) = <em>W</em><sub>1</sub><em>σ</em>(<em>W</em><sub>0</sub><em>x </em>+ <em>b</em><sub>0</sub>) + <em>b</em><sub>1</sub>

Use <em><sub>h </sub></em><sub>= 64 </sub>for the number of hidden units and choose an appropriate learning rate. Train the network until it reaches <sub>99% </sub>accuracy on the training data and provide a training plot (loss vs. epoch). Finally evaluate the model on the test data and report both the accuracy and the loss.

<ol>
<li><em>[10 points] </em>Let <em>W</em><sub>0 </sub>∈ R<em><sup>h</sup></em><sup>0</sup><sup>×<em>d</em></sup>, <em>b</em><sub>0 </sub>∈ R<em><sup>h</sup></em><sup>0</sup>, <em>W</em><sub>1 </sub>∈ R<em><sup>h</sup></em><sup>1</sup><sup>×<em>h</em></sup><sup>0</sup>, <em>b</em><sub>1 </sub>∈ R<em><sup>h</sup></em><sup>1</sup>, <em>W</em><sub>2 </sub>∈ R<em><sup>k</sup></em><sup>×<em>h</em></sup><sup>1</sup>, <em>b</em><sub>2 </sub>∈ R<em><sup>k </sup></em>and <em>σ</em>(<em>z</em>) : R → R some non-linear activation function. Given some <em><sub>x </sub></em><sub>∈ </sub>R<em><sup>d</sup></em>, the forward pass of the network can be formulated as:</li>
</ol>
F<sub>2</sub>(<em>x</em>) = <em>W</em><sub>2</sub><em>σ</em>(<em>W</em><sub>1</sub><em>σ</em>(<em>W</em><sub>0</sub><em>x </em>+ <em>b</em><sub>0</sub>) + <em>b</em><sub>1</sub>) + <em>b</em><sub>2</sub>

Use <em><sub>h</sub></em><sub>0 </sub><sub>= <em>h</em></sub><sub>1 </sub><sub>= 32 </sub>and perform the same steps as in part a.

<ol>
<li><em>[5 points] </em>Compute the total number of parameters of each network and report them. Then compare the number of parameters as well as the test accuracies the networks achieved. Is one of the approaches (wide, shallow vs. narrow, deeper) better than the other? Give an intuition for why or why not.</li>
</ol>
<strong>Using PyTorch: </strong>For your solution, you may not use any functionality from the torch.nn module except for torch.nn.functional.relu and torch.nn.functional.cross_entropy. You must implement the networks F from scratch. For starter code and a tutorial on PyTorch refer to the sections 6 and 7 material.

<h1>Using Pretrained Networks and Transfer Learning</h1>
<ol start="5">
<li>So far we have trained very small neural networks from scratch. As mentioned in the previous problem, modern neural networks are much larger and more difficult to train and validate. In practice, it is rare to train such large networks from scratch. This is because it is difficult to obtain both the massive datasets and the computational resources required to train such networks.</li>
</ol>
Instead of training a network from scratch, in this problem, we will use a network that has already been trained on a very large dataset (ImageNet) and adjust it for the task at hand. This process of adapting weights in a model trained for another task is known as <em>transfer learning</em>.

<ul>
<li>Begin with the pretrained AlexNet model from torchvision.models for both tasks below. AlexNet achieved an early breakthrough performance on ImageNet and was instrumental in sparking the deep learning revolution in 2012.</li>
<li>Do not modify any module within AlexNet that is not the final classifier layer.</li>
<li>The output of AlexNet comes from the 6th layer of the classifier. Specifically, model.classifer[6] = nn.Linear(4096, 1000). To use AlexNet with CIFAR-10, we will reinitialize (replace) this layer with nn.Linear(4096, 10). This re-initializes the weights, and changes the output shape to reflect the desired number of target classes in CIFAR-10.</li>
</ul>
We will explore two different ways to formulate transfer learning.

<ol>
<li><em>[15 points] </em><strong>Use AlexNet as a fixed feature extractor: </strong>Add a new linear layer to replace the existing classification layer, and only adjust the weights of this new layer (keeping the weights of all other layers fixed). Provide plots for training loss and validation loss over the number of epochs. Report the highest validation accuracy achieved. Finally, evaluate the model on the test data and report both the accuracy and the loss.</li>
</ol>
When using AlexNet as a fixed feature extractor, make sure to freeze all of the parameters in the network <em>before </em>adding your new linear layer:

model = torchvision.models.alexnet(pretrained=True) for param in model.parameters():

param.requires_grad = False

model.classifier[6] = nn.Linear(4096, 10)

<ol start="10">
<li><em>[15 points] </em><strong>Fine-Tuning: </strong>The second approach to transfer learning is to fine-tune the weights of the pretrained network, in addition to training the new classification layer. In this approach, all network weights are updated at every training iteration; we simply use the existing AlexNet weights as the “initialization” for our network (except for the weights in the new classification layer, which will be initialized using whichever method is specified in the constructor) prior to training on CIFAR-10. Following the same procedure, report all the same metrics and plots as in the previous question.</li>
</ol>
<a href="#_ftnref1" name="_ftn1">[1]</a> Given a dataset <em><sub>x</sub></em><sub>1</sub><em><sub>,…,x</sub></em><em><sub>n </sub></em><sub>∈ </sub>R<em><sup>d</sup></em>, a heuristic for choosing a range of <em><sub>γ </sub></em>in the right ballpark is the inverse of the median of all &nbsp;squared distances.
