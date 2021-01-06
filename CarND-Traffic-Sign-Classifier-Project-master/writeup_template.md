# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/examples_database.png "Some images from the database"
[image2]: ./examples/barChart_database.png "Bar cart of the training data"
[image3]: ./examples/example_grayscaling.png "Grayscaling"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34.799 images
* The size of the validation set is 4.410 images.
* The size of test set is 12.630 images
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43 traffic sign classes.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First you can see 12 randomly selected images with database code and description in the title:
![Images][image1]

Additionally, here is a bar chart showing how many images of each traffic sign in the training dataset are:
![Bar Chart][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because many images are very dark and grayscaling reduces color features. Additionally, the
further analytics are much more efficent with grayscaled images. My own experiencens and researches in the internet validate this.

As second and last step, I normalized the image data because this converts the RGB values of each pixel to float values between -1 and 1. This is useful because the furher analysis is much more comfortable with small numbers. I normalized the images by subracting 128 and dividing the difference by 128. The other commented out possibility, which was used in a quiz of this nanodegree, made my validation accuracy worse.

Here is an example of a traffic sign image before and after grayscaling:
![Grayscaled][image3]


I tried to generate additional data because as the bar chart above shows, some traffic signs have very less examples (e. g. 0 - Speed limit (20 km/h) compared to others (e. g. 2 - Speed limit (50 km/h). This may be caused by the frequency distribution in reality, but to regognize all signs equally good, I wanted to increase all data per traffic sign to the count of the most frequent traffic sign in the database (2010 images). This resulted in a much better validation accuracy at the beginning of training the modified LeNet model architecture. But the final result was always worse than the result without data augmentation. So I decided to leave the generation of additional data.

Nevertheless here is a short description how I tried data augmentation. The function increase_dataset in cell 6 would do the job. I used a random scaling factor between 0.8 and 1.2, a randomized rotation between -20 and 20 degrees and a random movement between -3 and 3 pixels in x and y direction. Even playing with these numbers brougt no better results.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gracscaled image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten			|												|
| Dropout 50 %   |       									|
| Fully connected		| outputs 120.        									|
| RELU				|        									|
| Dropout 50 %   |      									|
| Fully connected		| outputs 84.        									|
| RELU				|   
| Dropout 50 %   |      									|
| Fully connected		| outputs 43.        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, over all 150 epocs with a batch size of 128 and a learning rate of 0.0056. I experienced especially with the learning rate very often and came to the conclution that 0.0056 fits very good. The batch size got worse if I used another than 128, so I did not change this. The epocs can of course be more or less (around 50 epocs the validation accuracy seems already good enough) but more than 120 / 130 brought never a better result.

As mentioned above I also used 3 Dropout filters with rate of 50 %. Besides of that I experimented with mu (but every chage brought worse results) and sigma. For sigma I got the best result with 0.075 instead of 0.1.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I began with an architecture which overfitted overfit the training data (sometimes 99.6 %, whereas the validation accuracy was around 88 %. This is why I have added dropout layers between the fully connected layers. After that I spent a lot of time to adjust the hyperparameters epocs, batch_size, dropout rate, mu and sigma. Sometimes the validation accuracy was much worse than 93 %. I saved the model with the combination which had the highest validation accuracy.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: follows

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


