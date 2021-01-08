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
[image4]: ./examples/TrafficSignsFromTheInternet/0_Hoechstgeschwindigkeit20.jpg "0 - Speed limit (20km/h)"
[image5]: ./examples/TrafficSignsFromTheInternet/12_Vorfahrtstrasse.jpg "12 - Priority road"
[image6]: ./examples/TrafficSignsFromTheInternet/13_VorfahrtAchten.jpg "13 - Yield"
[image7]: ./examples/TrafficSignsFromTheInternet/13_VorfahrtAchten_verschneit.jpg "13 - Yield - with snow"
[image8]: ./examples/TrafficSignsFromTheInternet/14_Stop.jpg "14 - Stop"
[image9]: ./examples/TrafficSignsFromTheInternet/17_EinfahrtVerboten.jpg "17 - No entry"
[image10]: ./examples/TrafficSignsFromTheInternet/18_Achtung.jpg "18 - General caution"
[image11]: ./examples/TrafficSignsFromTheInternet/25_Baustelle.jpg "25 - Road work"
[image12]: ./examples/TrafficSignsFromTheInternet/2_Hoechstgeschwindigkeit50.jpg "2 - Speed limit (50km/h)"
[image13]: ./examples/TrafficSignsFromTheInternet/33_RechtsAbbiegen.jpg "33 - Turn right ahead"
[image14]: ./examples/TrafficSignsFromTheInternet/4_Hoechstgeschwindigkeit70.jpg "4 - Speed limit (70km/h)"
[image15]: ./examples/HowSureIsTheModel.png "How sure is the model?"
[image16]: ./examples/VisualizingNeuralNetwork.png "Visualizing neural Network"


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

As second and last step, I normalized the image data because this converts the RGB values of each pixel to float values between -1 and 1. This is useful because the furher analysis is much more comfortable with small numbers. I normalized the images by subtracting 128 and dividing the difference by 128. The other commented out possibility, which was used in a quiz of this nanodegree, made my validation accuracy worse.

Here is an example of a traffic sign image before and after grayscaling:
![Grayscaled][image3]


I tried to generate additional data because as the bar chart above shows, some traffic signs have very less examples (e. g. 0 - Speed limit (20 km/h) compared to others (e. g. 2 - Speed limit (50 km/h). This may be caused by the frequency distribution in reality, but to regognize all signs equally good, I wanted to increase all data per traffic sign to the count of the most frequent traffic sign in the database (2010 images). This resulted in a better validation accuracy at the beginning of training the modified LeNet model architecture. But the final result was always worse than the result without data augmentation. So I decided to leave the generation of additional data.

Nevertheless here is a short description how I tried data augmentation. The function increase_dataset in cell 6 would do the job. I used a random scaling factor between 0.8 and 1.2, a randomized rotation between -20 and 20 degrees and a random movement between -3 and 3 pixels in x and y direction. Even playing with these numbers brougt no better results.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | outputs 10x10x16      									|
| RELU					|	activation function											|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten			|												|
| Dropout 50 %   |       									|
| Fully connected		| outputs 120        									|
| RELU				| activation function       									|
| Fully connected		| outputs 84        									|
| RELU				| activation function   
| Dropout 50 %   |      									|
| Fully connected		| outputs 43        									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used AdamOptimizer, over all 150 epocs with a batch size of 128 and a learning rate of 0.00056. I experienced especially with the learning rate very often and came to the conclution that 0.000556 fits very good. And while training I reduced the learning rate from the 5th epoch on every 5th epoch about 5 %, so that the training epochs 1 to 4 have the original learing rate of 0.000556, epochs 5 to 9 have a learning rate of 0.000502 and so on until the last epoch 50 with a learning rate of   The batch size got worse if I used another than 128, so I did not change this. The epochs can of course be more or less, but 50 are enough as I saw. It may be that further epochs can increase the validation accuracy for some tenth percentpoints. This seemed to be not efficient.

I also added 2 Dropout filters with rate of 70 %. Besides of that I experimented with mu (but every chage brought worse results) and sigma (also no success).


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9 %
* validation set accuracy of 95.0 %
* test set accuracy of 93.6 %

I began with an architecture which overfitted the training data, sometimes 99.6 %, whereas the validation accuracy was around 88 %. This is why I have added 2 dropout layers between the fully connected layers. After that I spent a lot of time to adjust the hyperparameters epocs, batch_size, dropout rate, learning rate, mu and sigma. Sometimes the validation accuracy was much worse than 93 %. It was a long game of trial and error. I saved the model with the combination which had the highest validation accuracy. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eleven German traffic signs that I found on the web:

![0 - Speed limit (20km/h)][image4]
This image might be difficult to classify because the perspective and rotation angle might look like every other speed limit sign.

![12 - Priority road][image5]
This image should be no problem besides of the perspective.

![13 - Yield][image6]
Here the traffic sign is quite small compared to the other images and there is a road sign above.

![13 - Yield - with snow][image7]
The traffic sign is big enough but parts of the sign are hidden by snow.

![14 - Stop][image8]
Here the perspective might be a problem, on the other hand there is no resembling traffic sign.

![17 - No entry][image9]
This sign should be easy to classify.

![18 - General caution][image10]
Here is the difficulty an additional sign below the interesting sign

![25 - Road work][image11]
This sign is ab bit blurry and the perspective could make the classifying harder.

![2 - Speed limit (50km/h)][image12]
This should be no problem although there are many resembling speed limit signs.

![33 - Turn right ahead][image13]
Here it is again the perspective, on the other hand the sign is good to see.

![4 - Speed limit (70km/h)][image14]
This should be no problem although there are many resembling speed limit signs.





#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0 - Speed limit (20km/h)      		| 0 - Speed limit (20km/h)   									| 
| 12 - Priority road     			| 12 - Priority road										|
| 13 - Yield					| 13 - Yield											|
| 13 - Yield - with snow	      		| 13 - Yield				 				|
| 14 - Stop		| 14 - Stop     						|
| 17 - No entry		| 17 - No entry      						|
| 18 - General caution		| 1 - Speed limit (30km/h)      						|
| 25 - Road work		| 25 - Road work     						|
| 2 - Speed limit (50km/h)	| 2 - Speed limit (50km/h)     						|
| 33 - Turn right ahead		| 33 - Turn right ahead	    						|
| 4 - Speed limit (70km/h)		| 4 - Speed limit (70km/h)      						|



The model was able to correctly guess 10 of the 11 traffic signs, which gives an accuracy of 90.9 %. This compares favorably to the accuracy on the test set of 93.6 %.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th cell of the Ipython notebook.

Here is a bar chart for all eleven images. On the left side is a picture with the description as title, on the right side is a bar chart of the top five softmax probabilities of the model prediction. We can see, that the model was often very sure to see the correct image. Nevertheless sometimes there are other rows (e.g. the third image - Yield - here the model also thought wich about 28 % that this is an "ahead only" sign. But with more than 70 % it guessed right).

The one misclassified image is interesting. Here the model was for more than 60 % sure to see a 'Speed limit (30km/h)' sign. But with each less than 10 % it thought of another possibility. The fifth possibility would have been the correct answer.

![How sure is the model?][image15]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![Visusalizing neural network][image16]

It is hard to see. The feature maps should look for slightly different edges, but here nearly all are activated by the diagonal line from top left to the middle down of the Yield sign.
