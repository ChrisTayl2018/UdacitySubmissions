# **Traffic Sign Classifier** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Visualisation/Data_Visualisation.png "Visualisation"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./German_Traffic_Sign_data/German_Traffic_Signs/examples/minimum_30kmh_Speed_required.png "Traffic Sign 1"
[image5]: ./German_Traffic_Sign_data/German_Traffic_Signs/examples/priority_road.png "Traffic Sign 2"
[image6]: ./German_Traffic_Sign_data/German_Traffic_Signs/examples/right_turn_ahead.png "Traffic Sign 3"
[image7]: ./German_Traffic_Sign_data/German_Traffic_Signs/examples/wild_animals.png "Traffic Sign 4"
[image8]: ./German_Traffic_Sign_data/German_Traffic_Signs/examples/stop_and_give_way.png "Traffic Sign 5"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is: 43 Classes

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The data had some simple yet effective pre-processing techniques used before being input into the the CN. It is first important to note in this submission that there were multiple attempts made using both RGB and grayscale + normalised images as the training, validation and testing data on the CNN LeNet Architecture. Briefly explained are the seperate techniques used for pre-processing the data:

##### Traffic_Sign_Classifier_Model1

- Data was shuffled using a imported shuffle function, Data was RGB so no normalisation or gray scaling was implemented on the images
                                 

##### Traffic_Sign_Classifier_Model1_Gray: 

- Data was shuffled using imported shuffle function. RGB Values were put through a 'reshape_images(imgs)' function in which carried out normalisation (scaling all values between 0 -> 1), gray scalling and reshaping the images in which the output image was a 32 X 32 x 1. Specifically the output of the function was a numpy array (32,32,1). 

##### Traffic_Sign_Classifier_Model2: 

- Data was shuffled using imported shuffle function such as the example 'X_train, y_train = shuffle(X_train, y_train)', Data was RGB so no normalisation or gray scaling was implemented on the images

The code for this can be found in the cells 9 and 10 in all three of the models. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet architecture used within the course was ported over into this project and utilised. This was due to my understanding of CNN's and it was felt best to utlise a model that was proven to work on other datasets. Upon implementing the LeNet architecture on the german sign dataset it was noted that the training accuracy was only ever reaching about 0.7 or 70% accuracy. This was seen as unsatisfactory so some changes were made to the vanilla architecture as suggested throughout the training course and online. The changes included an added CNN layer, a added fully connected layer and finally a drop out layer. After adding the dropout layer as an added regularization technique, significant imporvements in the accuracy were noted with model achieving 90% accuracy. 

The final model used for the submission was Traffic_Sign_Classifier_Model2:


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 3x3	    | etc.      									|
| RELU					|												|
| Convolution 3x3	    | etc.      									|
| Max pooling	      	| 2x2 kernel size, 2x2 stride, valid padding	|
| Flatten   	      	| 2x2 stride,  outputs 256      				|
| Fully connected		| Input 256, Output 120							|
| RELU					|												|
| Dropout				|keep_prob =0.5									|
| Fully connected		| Input 120, Output 100							|
| RELU					|												|
| Fully connected		| Input 100, Output 84 							|
| RELU					|												|
| Fully connected		| Input 84 , Output 43 							|
						|												|
 
 
 The code for this is found in cell 13. 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The training parameters used to train the model were as follows: 

Epochs: 75
Batch Size: 250
Learning Rate: 0.0009
Loss Function: Softmax Cross Entropy
Optimisation Function: Adam Optimisation 
mu: 0
sigma: 0.1
dropout keep probability:0.5

This was defined in celld 14 and 17

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.966
* test set accuracy of 0.947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
RE: a standard vanilla LeNet(x) architecture was chosen and was chosen based of the previous task with the MNIST dataset. The model was already proven to be functional so it was adopted with minimal changes. 

* What were some problems with the initial architecture?
RE: accuracy was a bug problem. The current architecture was not acchieving over mid 70% accuracy in which was not meeting the requirements in section 4 > 0.93. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

RE: The changes included an added CNN layer, a added fully connected layer and finally a drop out layer. After adding the dropout layer as an added regularization technique, significant imporvements in the accuracy were noted with model achieving 94% accuracy. The convolutional layer was added in a process of trial and error in which the output was compared to see if the relative accuracy was decreasing. The dropout layer was added in as suggested to prevent over fitting 

I also found lowering the learning rate, varying the epoch and batch size significantly increased the accuracy. The mean (mu) and variance (sigma) values were also varied slightly in a trial and error attempt how ever it was found that varying the variance slightly even by 0.01 that it significantly decreased the accuracy of the network. These parameters were best thought left alone. 




### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because:

- Additional image preprocessing techniques were needed on these images as they were all of different sizes and shapes. 
- The backgrounds of the image were all  different to the training images 
- The contrast as well as transparency was different. These images were of actual icons/symbols in comparison to the real world images captured in the datasets. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Unfortunately the model did not predict the signs correctly. After multiple iterations I was unable to find the problem. I suspect this to be a problem with either the class labels or perhaps the fact I used RGB values. When the images were loaded in using the opencv2 tool it changed the colors of the image and hence might have affected the classifier. I suspect how ever that it was more due to a minor bug in the class labels how ever I was unable to find it in time given the project is already overdue. 

As an attempt I did try loading some fresh images in using matplotlib instead how ever this was loading 4 dimmensional data (an example of image.shape[1] returned (120,120,4)). I was not able to handel 4 dimmensional data and hence did not progress. Given the model was trained up to relatively high accuracy > 93%, I believe there to be a minor problem in my code somewhere and if had more time I would inspect and debug. 

Looking at some of the results: 

TopKV2(values=array([[  9.91776943e-01,   6.79543708e-03,   1.33401330e-03,
          3.70350463e-05,   2.53952112e-05],
       [  9.99659657e-01,   3.21102329e-04,   1.44958622e-05,
          2.51540496e-06,   1.12776866e-06],
       [  9.99999881e-01,   7.18727833e-08,   4.45239375e-16,
          4.66586817e-18,   1.87111628e-19],
       [  6.98998690e-01,   1.52099773e-01,   1.48770913e-01,
          1.30593355e-04,   5.46909540e-09],
       [  4.58547890e-01,   1.89743161e-01,   1.71790764e-01,
          6.36151880e-02,   4.15636152e-02],
       [  9.86362517e-01,   1.19400863e-02,   1.07716466e-03,
          3.12017277e-04,   2.02879834e-04]], dtype=float32), indices=array([[ 1, 31,  0,  4,  7],
       [ 1,  4,  0,  5,  2],
       [40, 39, 38, 37,  2],
       [25, 38, 34, 10, 29],
       [40, 36,  5, 33, 25],
       [33, 34, 35, 36, 13]], dtype=int32))



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


This task was left uncompleted how ever will need to be comepleted towards the end of the semester after I catch back up. A suspected problem in the neural network prediction stage on new images has prevented me from obtaining the model certainty 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?'

The optional task was left uncompleted intentionally. 


