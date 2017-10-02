# **Traffic Sign Recognition** 

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

[bar_chart]: ./md_assets/bar_chart.png "Bar Chart"
[rotation]: ./md_assets/rotation.png "Rotation"
[model_arch]: ./md_assets/model_arch.png "Model Arch"
[traffic_sign_1]: ./traffic_signs/0.png "Traffic Sign 1"
[traffic_sign_2]: ./traffic_signs/10.png "Traffic Sign 2"
[traffic_sign_3]: ./traffic_signs/24.png "Traffic Sign 3"
[traffic_sign_4]: ./traffic_signs/27.png "Traffic Sign 4"
[traffic_sign_5]: ./traffic_signs/38.png "Traffic Sign 5"
[conv_1]: ./md_assets/conv1.png "Conv 1"
[conv_2]: ./md_assets/conv2.png "Conv 2"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
It is a bar chart showing the class distribution of the training data.

![alt text][bar_chart]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

* I normalized the training data with min max normalization.
Keeping pixel magnitude below 0.5 will prevent activations being too large.
* I augmented the training set by adding extra data which rotates original images
clockwise and counter clockwise by 10 degrees. Below is an example:

![alt_text][rotation]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model (LeNet) consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| ELU                   |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120        							|
| ELU                   |                                               |
| Fully connected		| outputs 84        							|
| ELU                   |                                               |
| Dropout               | keep prob 0.5                                 |
| Fully connected		| outputs 43        							|
| Softmax				|        									    |

![alt_text][model_arch]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used following setups:
* optimizer: Adam
* batch size: 128
* number of epochs: 50
* learning rate: 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.954
* test set accuracy of 0.943

* What was the first architecture that was tried and why was it chosen?

  I first implemented a softmax regression model since it is good for fast prototyping
and it does not require GPU resources.

* What were some problems with the initial architecture?

  Model is too simple and underfits the data.

* What architecture was chosen?

  The final model is similar to LeNet's architecture with adjusted input and output dimensions.
  It also uses ELU for faster convergence and dropout to reduce overfitting.

* Why did you believe it would be relevant to the traffic sign application?

  LeNet is proven to be successful in digit recognition tasks.
It should be promising for recognizing road signs as well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

  Both validation accuracy and test accuracy passes the base line (0.93).
 Although the model definitely has room for improvement,
 but it is already a good enough POC for using CNN on traffic sign dataset.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][traffic_sign_1] ![alt text][traffic_sign_2] ![alt text][traffic_sign_3] 
![alt text][traffic_sign_4] ![alt text][traffic_sign_5]

First image might be hard to classify since part of the sign is hidden.

Third image might be hard to classify since it is very dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h)  | Speed limit (120km/h)  						| 
| No passing over 3.5t  | No passing over 3.5t							|
| Road narrows on right	| Bicycles crossing		                    	|
| Pedestrians	        | Pedestrians					            	|
| Keep right	        | Keep right     		                        |

The model gives 60% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Image: Speed limit (20km/h)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99          		| Speed limit (120km/h)   						| 
| 3.47e-06    		    | Speed limit (80km/h) 							|
| 2.03e-06				| Speed limit (20km/h)                   		|
| 1.47e-08	      		| Speed limit (70km/h)					 		|
| 6.63e-11				| Keep right                 					|

Image: No passing over 3.5t

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00e+00         		| No passing over 3.5t10,  7, 25, 12, 42  	    | 
| 2.14e-26    		    | Speed limit (100km/h) 				        |
| 1.42e-26				| Road work                   		            |
| 3.15e-28	      		| Priority road					 		        |
| 7.35e-29				| End of no passing over 3.5t              		|

Image: Road narrows on right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98          		| Road narrows on the right  					| 
| 1.38e-02    		    | Bicycles crossing 							|
| 3.22e-03				| Speed limit (120km/h)                   		|
| 2.12e-03	      		| Double curve					 	        	|
| 3.14e-04				| Children crossing                 		    |

Image: Pedestrians

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99          		| Pedestrians             						| 
| 1.39e-05    		    | General caution 						    	|
| 2.82e-07				| Road narrows on right                   		|
| 1.56e-07	      		| Speed limit (30km/h)					 		|
| 7.58e-10				| Speed limit (20km/h)                 			|

Image: Keep right

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         	    	| Keep right                       		    	| 
| 6.37e-33    		    | Dangerous curve to the right 				    |
| 0.00e+00				| Speed limit (20km/h)                   		|
| 0.00e+00	      		| Speed limit (30km/h)					 		|
| 0.00e+00				| Speed limit (50km/h)                 			|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][conv_1]

Feature map from first convolution layer shows some patterns of the shape of traffic signs.

![alt text][conv_2]

Since second convolution layer only has feature map of 5x5, it is very hard to tell what patterns it learned.
