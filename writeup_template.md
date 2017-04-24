**Traffic Sign Recognition**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/data_visualization.png "Visualization"
[image2]: ./examples/color_train_set.png "Color Images"
[image3]: ./examples/gray_train_set.png "Grayscaling"
[image4]:./examples/balanced_classes.png "Resampled data distribution"
[image5]: ./examples/googlenet_img.png
"Googlenet Structure"
[image6]: ./examples/caution.jpg "Traffic Sign 1"
[image7]: ./examples/speed.jpg "Traffic Sign 2"
[image8]: ./examples/stop.jpg "Traffic Sign 3"
[image9]: ./examples/turn.jpg "Traffic Sign 4"
[image10]: ./examples/yield.jpg "Traffic Sign 5"
[image11]: ./examples/softmax_prob.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/usryokousha/Traffic_Classifier_Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34,799
* The size of the validation set is: 4,410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32,32,3)
* The number of unique classes/labels in the data set is: 43

####2. Include an exploratory visualization of the dataset.

Here is the class distribution of the Training Set data

As you can see the class bin sizes are unbalanced.  If left unchanged, this imbalance will result in lower performance in minority classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially I started with the LeNet-5 (Lecun et Al) using the color images in the training set.  I made sure to normalize them around the center of the RGB gamma curve (0-255) or norm_val = int(input_val / 127.5 - 1).

After running the model with the color training data, I ended up achieving around 90% validation accuracy using dropout with an retain probability of 50%.

It seemed only natural to attempt running the model with a grayscale training set.  This turned out to be true with a resulting validation accuracy of 93.7%


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image3]

To balance the class data in the Training set I looked into a number of options.  I built my own random resampling algorithm that generates a mask to be applied to the original training data.  I had trouble getting it to completely balance the dataset because of rounding errors and decided to use imbalanced-learn's random oversampling algorithm.  After running the the resampled Training data through the LeNet-5 model, I was getting around 95% validation accuracy.

I thought I could go a step further and add synthetic data and chose to go with the SVM SMOTE (synthetic minority oversampling) algorithm.  imbalanced-learn's package does not support a mult-class SMOTE algorithm however.  I modified the masking code originally used to resample the dataset to run SMOTE through each class and generate resampled data.

Since SMOTE works by changing the L1 Norm vector's length, I thought it wouldn't significantly harm the data if I used a value for k of 2.  However, after using the resampled SMOTE data the performance was similar to that of the randomly oversampled data (ROS).  Using both SMOTE and ROS my training set expanded to 86,430 images (248% increase).

![alt text][image4]  

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was based on the "inception" layers found in the paper "Going Deeper with Convolutions" by Szegedy et Al (https://arxiv.org/abs/1409.4842)

This model adds many convolutional layers without sacrificing intense computation time and delivers very good performance in terms of generalization and avoiding overfitting.

I built my convolutional layers emulating the Googlenet paper's model as closely as I could while trying to maintain efficient code. The "inception" layers in the model were built using the following layers.

![alt text][image5]

In addition, to achieve correct dimensionality of my data and save computational resources, my second convolutional layer also included an initial 1x1 reduction layer before my 3x3 filter convolution.

Main Model

| Main Layers         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Convolution 1x1 | 1x1 stride, same padding, outputs 28x28x18|
| RELU | |
| Convolution 3x3|1x1 stride, same padding, outputs 28x28x36|
| Inception 1| See Inception table, outputs 28x28x48|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48				|
| Inception 2	| See Inception table, outputs 14x14x64  |
| Max pooling	      	| 2x2 stride,  outputs 7x7x64				|
| Flatten			| 7x7x64 --> 3,136 |
| Fully Connected 1| outputs 3136x448|
| RELU | |
| Dropout	|	50% keep probability|
| Fully Connected 2| outputs 448x43|

| Inception Layers         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| Previous layer  							|
| Input connected layers | |
| Convolution 1x1 (direct)| 1x1 filter width, 1x1 stride, same padding, output depth: 1/3 input_depth	|
| Convolution 3x3 reduction| 1x1 filter width, 1x1 stride, same padding, output depth: 1/2 input depth|
| Convolution 5x5 reduction| 1x1 filter width, 1x1 stride, same padding, output depth: 1/12 input depth|  
| Max Pooling | 3x3 filter width, same padding, 1x1 stride|
| RELU | RELU present after all input connected layers|
| Reduction connected layers | |
| Convolution 3x3| 3x3 filter width, same padding, outputs: 2/3 input depth|
| Convolution 5x5| 5x5 filter width, same padding, outputs: 1/6 input depth|
| Max Pooling 1x1 Convolution|1x1 filter width, same padding, outputs: 1/6 input depth|
| Concatenation | concatenation of all reduction connected layers and 1x1 convolution|
|RELU | final activation after concatenation|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The initial parameters for my model "mini_googlenet" were chosen to prevent overfitting and to achieve greatest efficiency.

| Model Parameters         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Batch Size | 128|
| Epochs| 10 |
| Learning Rate | 0.001|
| Dropout rate (training)| 0.5|
| Dropout rate (validation)|1|

I tried SVM but stuck with the ADAM optimizer since it finds the error minimum most quickly and thus results in a faster model.  With the inception implementation my model runtime was around 2 hours on AWS.  I also played around with L2 regularization and got an improvement in validation accuracy, but worried about an increase in training time and pulled it out of my final model.

I also tried applied an exponential decay to my learning rate for each epoch, but could not balance speed with overall maximum validation accuracy.



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 98.2%
* test set accuracy of 95.9%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet-5 was the initial architecture chosen, but after resampling the training data, grayscaling, adding dropout and L2 regularization I found I couldn't get over 95-96% validation accuracy.  I increased the number of epochs to 30 and tried lowering the learning rate, but couldn't seem to get much higher than 96% validation accuracy.  I had a feeling that adding layers would help, and using inception I was able to show a dramatic improvement.
* What were some problems with the initial architecture?
The initial architecture only uses two convolution layers which limits the types of features it extracts.  Training with a much deeper model seems to generalize better.
* How was the architecture adjusted and why was it adjusted? From the outset I was afraid of overfitting, and I quickly implemented dropout in the fully connected layers of the model.  I played around with different probabilities, but stuck with the 50% value.  I trained the model at lower learning rates, and got slightly improved validation and test accuracies. L2 regularization seemed to close the gap even further between the validation accuracy and the test accuracy.
* Which parameters were tuned? How were they adjusted and why?
I believe I answered this question above.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
This is an image recognition problem and different receptive field sizes gives the model greater sensitivity to a variety of features.  Especially with the added depth of two inception layers there is a tendency for the model to fire more for some pronounced features.  Dropout helps the model to bring out more of the subtle features as well.

If a well known architecture was chosen:
* What architecture was chosen? googlenet (Inception)
* Why did you believe it would be relevant to the traffic sign application? This is an image recognition model and additional convolutional layers tend to improve model accuracy for vision tasks.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  Validation accuracy gives you a good idea of how your model is performing on the data you are training. As the validation accuracy climbs the weights in the model are being adjusted so as to reduce error.  If the test accuracy is close to your validation accuracy, you know your model is not specific to the training data and works well on new data as well.
Very high training accuracy might indicate overfitting since it is simply evaluating the training data with the same model it was trained on.  In general it is better to use the validation accuracy as a gauge of training performance since it is done with data set aside for evaluation.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| General Caution      		| Bicycle Crossing   									|
| Speed Limit 30 km/h    			| Road Work 										|
| Yield					| Yield											|
| Stay Right	      		| No Entry				 				|
| Stop			| Stop      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This is surprising given that the data was formatted the same way and the signs complexity was not too high.  The resolution reduction performed on the images did not seem to degrade their features too significantly either.

The sign in several of the images was small in relation to the dimensions of the image.  Since the images were processed in grayscale, this may make it difficult to discriminate between certain colored signs. In the case of the 3 errors, the predicted label was for a sign that had the same shape as the actual sign in the image.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The prediction results are located in my Ipython notebook but the following image below is a graph capturing the different probabilities.

Interestingly, the softmax probabilities for the 30 km/h speed limit sign and general caution sign were 1.0 for bicycle crossing and road work respectively.  Despite fixation on the shape of the sign, it is odd that other candidates did not appear.  The final incorrect label was the stay right sign, which does show one of the candidates as turn right ahead.
![alt text][image11]  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
