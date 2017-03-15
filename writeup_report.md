
# Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_and_notcar1.png
[image2]: ./output_images/HOG_Features.png
[image3]: ./output_images/original_sliding_windows1.png
[image4]: ./output_images/mod_sliding_windows1.png
[image5]: ./output_images/boxes_heatmaps1.png
[image6]: ./output_images/threshold_heatmaps1.png
[image7]: ./output_images/scaling.png
[video1]: ./test_output1.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook in the functions `get_hog_features()` and `extract_features()`. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `skimage.hog()` parameters and color spaces. I eventually settled on these following parameters since they gave me the best results: `color_space: YCrCb`, `orientations = 9`, `pix_per_cell = 8`, and `cell_per_block = 2`. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is contained in the 6th code cell of the IPython notebook. I trained a linear SVM using hog features along with spatial and color histogram features. I used the following parameters for training:  `color_space: YCrCb`, `orientations = 9`, `pix_per_cell = 8`, `cell_per_block = 2`, `hog_channel = 'ALL'`, `spatial_size = (32, 32)`, and `hist_bins = 32`. Sharper images and better vehicle detection was achieved by using this combination of parameters. I noticed that increasing the number of orientations gave better training results but it meant using more memory so I just settled with leaving the number of orientations at 9. A scaler called `StandardScaler()` was used to normalize the features after extraction to a mean of zero. This is shown below: 

![alt text][image7]

Normalizing ensures that individual features or sets of features do not dominate the response of the classifier. The data was split into training and test sets with the test size at 20% of the total data. The resulting feature vector length was 8460. Using the YCrCb color space and all hog channels instead of one resulted in better performance. The accuracy I achieved by training a linear SVM was 99%. Even though I achieved a high accuracy I still experienced seeing false positives in the images. I then saved the result from the classifier to a pickle file so that I can load it easily without having to train again. 

### Sliding Window Search
 
#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I first implemented the original sliding windows method described in the course lessons. The code for this technique is contained in the 4th code cell of the IPython notebook in the functions `slide_window()` and `search_windows()`. The implementation of the method on the test images is found in the 11th code cell. A car is searched in a specified area of the image by sliding a window or multiple windows across the image with an overlap. The window has a window size and overlap associated with it. Each window that is used for sampling is rescaled to 64x64 so that the classifier can make a prediction on the selected region. Features extracted from the image patch that is formed from each window are fed to the classifier to detect whether the patch is a car or not. I decided to use windows with multiple sizes and an overlap of 0.5. The windows sizes I used were 96x96, 128x128, and 192x192. I chose to search the bottom half of the image to focus on just the cars and exclude the scenery like the trees. By only searching the bottom half of the image, some false positives would be removed. Here are some example images showing the use of this method: 

![alt text][image3]

I decided to abandon this method and use the technique described in the Udacity video tutorial for this project since I got better overall performance in detecting cars and improved processing time. The video can be found in this link [Video Tutorial](https://www.youtube.com/watch?v=P2zwrTM8ueA&t=3434s&list=PLAwxTw4SYaPkz3HerxrHlu1Seq8ZA7-5P&index=5). The code for this modified technique is contained in the 12th code cell in the function `find_cars()`. The implementation of the method on the test images is found in the 14th code cell. Instead of extracting hog features for each window as was done in the previous method I was using, the hog features are only extracted once over the entire image that is being sampled and then the corresponding patch of the image is subsampled at different scales. This method improved the performance of my pipeline. I decided to use multiple scales to zone in on the cars and improve detectability. I used the following scales: 0.9, 1.2, 1.4, and 1.6. I also focused on the right lower half of the image to apply the modified window sliding search for the cars. Focusing on this region decreased the processing time of my pipeline and also discarded the cars that were on the left side of the highway medium in the image. I noticed that if I used small scale sizes, the processing time would increase and more false positives were being recorded. Using large scale sizes decreased processing time but it also introduced large windows that brought errors in the detections. A tradeoff had to be made between the quality of detections and processing speed and I found a middle ground of scale values that provided reasonable results for detecting cars. Furthermore, I used a `cells_per_step` value of 2. The `cells_per_step` are the step counts of the windows. This value is used instead of the overlap value that was used in the previous sliding windows method that I discussed. By adjusting the steps I observed that a low step value increased the number of false positives in the image but improved the detection of cars. Increasing the step value eventually caused the pipeline to miss some car detections. A step value of 1 gave me the best results in terms of detecting cars but I could not remove the false positives. I eventually settled on a step value of 2. The last thing I adjusted was decreasing the resizing of the image patch from 64x64 to 32x32. This reduced the processing time of my pipeline. I tested using both sizes and I got identical results so I ended up using a resizing of 32x32.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Searching on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the features vector gave me the best results. As mentioned above I also focused on the right lower half of the image, used a step value of 2, and resized the image patch to 32x32 to optimize the performance of my pipeline. To further optimize my pipeline I applied the use of a heatmap with thresholding to remove false positives. I will discuss this filtering technique in a later section below. Here are some example images showing the use of the modified sliding windows method:                 

![alt text][image4]

The images above do not take into account the correction for false positives which I will discuss in a later section below by using a heatmap with thresholding. 

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_output1.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I first recorded the positions of positive detections in each frame of the video. From the positive detections, I created a heatmap. I then used `scipy.ndimage.measurements.label()` to identify individual hot spots in the heatmap.  I then assumed each hot spot corresponded to a vehicle. I constructed bounding boxes to cover the area of each hot spot detected. The code for this implementation is found in the 15th code cell. Below are example images showing the bounding boxes and the corresponding heatmaps:

![alt text][image5]

Not all the false positives were able to be removed by just using the heatmap and assuming that each hot spot corresponded to a vehicle. I decided to threshold the individual heat maps to better identify cars. After applying the threshold, bounding boxes were constructed again to cover the area of each resulting hot spot detected. Below are example images showing the bounding boxes and the corresponding heatmaps after thresholding:

![alt text][image6]

To smooth out the bounding boxes and make them less jittery during video processing, I decided to keep track of the heatmaps from the last 10 frames and average them. The resulting heatmap from averaging is then thresholded to identify cars and construct bounding boxes around them. The last 10 heatmaps were kept in the `Track()` class found in the 16th code cell. The implementation to smooth out the bounding boxes is found in the 17th code cell in the function `process()` that is used to process each image from the project video. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problems I faced were removing false positives and reducing processing time. The approach I took was through trial and error to find optimal values for the scales and steps while maintaining quality detections and low processing time. I also tried different thresholding values for the heat maps and elected on using a threshold of 3.2. Using a higher threshold would lead to lower quality of detections for cars. In addition, I optimized the search area of the image to lower the processing time by just focusing on the lower right side of the image where cars were likely to be found in the video. The pipeline would likely fail if there are different shadows/lighting in the images that would make it hard for the classifier to detect. Bad weather conditions such as rain or snow would make my pipeline fail. To make my pipeline more robust I could try using a different classifier to better detect cars or combine the current classifier with a convolutional neural network. Using a deep learning approach could help improve the classification of cars. Using a larger data set for training may lead to higher accuracy of car detections. Furthermore, I can work on designing a better algorithm to smooth out the bounding boxes over successive frames to make them less jittery. Lastly, I would like to work on improving the speed of my pipeline and decreasing the processing time even further to get faster results. 
