#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_summary.png "Model Visualization"
[image2]: ./examples/center_drive_sample.jpg "Center"
[image3]: ./examples/left_recover.jpg "Recovery Image"
[image4]: ./examples/right_recover.jpg "Recovery Image"
[image5]: ./examples/turn_sample.jpg "Turn Image"
[image6]: ./examples/rgb_image.png "Normal Image"
[image7]: ./examples/flip_image.png "Flipped Image"
[image8]: ./examples/yuv_image.png "YUV Image"
[image9]: ./examples/crop_image.png "Crop Image"
[image10]: ./examples/distrubition.png "distribution Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run_30mph.mp4 is the recorded video while setting the throttle to 30 mph. The overshooting is obvious but overall the car can stay on the track properly.
* run_20mph.mp4 is the recorded video while setting the throttle to 20 mph which is more smooth.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is adapted form the Nvidia neural network(code line 111-132). The data fed in is transferred to the YUV color space first(code line 57), then cropped top and bot part of the image(code line 113). And the data is normalized in the model using a keras lambda layer(code line 115)


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 118 etc.). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 31). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving on 1st and 2nd tracks, recovering from the left and right sides of the road, augmented left and right images, turnning in the curve lane data.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first step was to use a convolution neural network model similar to the nvidia NN. I thought this model might be appropriate because it has been tested on the similar dataset and has good performance. In this project, I just use the CNN to predict the steer angle so I think this model is complex enough.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding some dropput layers so that the validation loss can be reduced. Then I added keras checkpoints and earlystop callback to the model so that I can always have the best model I trained without worrying about choosing the epoch number carefully.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, for example the car kept falling into the water in first turn. To improve the driving behavior in these cases, I revisited the way I preprocess the data and add more tricks there and also add some patience value to the earlystop callback to make sure the model is fully converged.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 111-143) consisted of a convolution neural network with the following layers and layer sizes 

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one CW and CCW using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when there is some disturbance. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]

Besides, in order to handle turning better. I recorded more data while car is turning.

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help generating more data. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 26360 number of data points. I then preprocessed this data by coverting them to YUV color space and cropping some usless top and botoom part.

![alt text][image8]

Also the top and bot part is cropped to help remove useless information.

![alt text][image9]

Besides, in order to compensate the bias caussing by two many samll steer angle data points. I randomly dropped 85 percentage of the data points of which the steer angle is less than 0.02 rad to get a better steer measurement distribution.
Here is an exmple of the sample data provided by carnd distribution before and afer this random drop parocess.

![alt text][image10]

Meanwhile, a correction of 0.23 rad is applied the left and right images so that we can use them as trainning data input as well.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I added chekpoints and earlystop callback to the git generator so I am not to worry about the epoch number choosing. Based on the experiments, model usually converged at 5-10 epoches. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Thoughts and tips
1. The autonomous driving performance is highly related to the how the trainning data is generated.
2. 0.2 - 0.25 rad correction works well for left and right images augmentation.
3. Nvidia model works good but some dropout layers are needed to prevent the overfit.
4. Make sure preprocess the images same way as the model.py in the drive.py.
5. It is more difficult to handle high speed autonomous driving while testing the model in simulator.
