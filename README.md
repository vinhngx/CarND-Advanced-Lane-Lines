## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_img_example]: ./examples/undistorted_img_example.png "Undistorted"

[image2]: examples/undistorted_test1.png "Test image"
[image3]: ./examples/undistorted_test1_binary.png "Binary Example"
[image4]: ./examples/undistorted_test1_warped.png "Warp Example"
[image4b]: ./examples/binary_test1_warped.png "Binary Warp Example"
[image5]: ./examples/window_based_lane_detection.png "Window based lane detection"

[image6]: ./examples/pipeline_single_final.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the IPython notebook [camera_calibration.ipynb](./camera_calibration.ipynb).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![undistorted_img_example][undistorted_img_example]

### Pipeline (single images)

The pipeline for single images is detailed in the Jupyter notebook
 [pipeline-single-image.ipynb](./pipeline-single-image.ipynb).

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color transform and gradient thresholds to generate a binary image. This step is detailed in Step 2 of the Jupyter notebook [pipeline-single-image.ipynb](./pipeline-single-image.ipynb). Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is detailed in Step 3 of the Jupyter notebook [pipeline-single-image.ipynb](./pipeline-single-image.ipynb).

The source and destination points are defined as below:

```python
#Four source cordinates
src = np.float32(
    [[254, 700],
     [595, 449],
     [688, 448],
     [1062, 693]
    ])
#Four desired cordinates
dst = np.float32(
    [[254, 700],
     [254, 0],
     [1062, 0],
     [1061, 693]
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 254, 700      | 254, 700     | 
| 595, 449      | 254, 0       |
| 688, 448      | 1062, 0      |
| 1062, 693     | 1061, 693    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]
![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for finding lane lines is detailed in Step 4 of the Jupyter notebook [pipeline-single-image.ipynb](./pipeline-single-image.ipynb). We employ a sliding window and histogram based method. This step is applied on a binary warped image. The vertical dimension of the image is divided into 9 equi-height sections. 
The image is devided into two halves: the left lane and right lane are detected seperately. 
The base points for start searching the lane is the peak location of the histogram. 

Starting from the bottom section of the image, for each window centred at the base point, if the number of pixel is larger than the threshold, they are determined to form part of the respective lane lines. 
Finally, second order polinomials are fitted to the left and right lanes seperately.
The outcome of this step for the above test sample is demonstrated in the below image:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating road curvature and position of the vehicle is detailed in Step 5 of the Jupyter notebook [pipeline-single-image.ipynb](./pipeline-single-image.ipynb).

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Step 6 of the Jupyter notebook [pipeline-single-image.ipynb](./pipeline-single-image.ipynb) demonstrates the final lane area detection result super-imposed on the original image. 

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code for processing videos is contained within the Jupyter notebook [pipeline-video.ipynb](./pipeline-video.ipynb).

Here's a [link to my video result](./project_video_proccessed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach carried out in this project performs quite well on the project video. However, it did not perform that well on the more chalenging videos (see, for example, [challenge_video_proccessed.mp4](./challenge_video_proccessed.mp4)). 

In order to improve the lane detection results, I have tried the following alternatives (detailed in pipeline-video-alternative.ipynb):

- Define a region of interest using the approach in the first lane detection project:

```python
#Define a region of interest
vertices = np.array([[(0,Y),(0.45*X, 0.55*Y), (0.55*X, 0.55*Y), (X,Y)]], dtype=np.int32)
selected_img = region_of_interest(result, vertices)
```

- Combine hough transform for lane line detection with a window based approach

- Memorize the base location of the lanes in the current frame and use that as a starting search point for the next frame. This alternative approach is implemented in the notebook "pipeline-video-alternative.ipynb". 

- Combining Hough transform with histogram based approach (pipeline-single-image-Hough.ipynb)

Despite these modification, it is still challenging to detect lanes in the challenging videos.

The approach presented in this project relies heavily on a hand-designed pipeline, with expert understanding of the color space, and some heuristical rules (such as binary thresholding, histogram threshold to detect lane pixels). Learning-based approaches using human-annotated lane lines might improve the existing approach, as well as resolving corner cases (such as left lane appearing in the right-half of the image (due to vehicle veering off road). 


### 2nd attempt

The approach carried out in the first submission, although doing well on most of the frames, seriously mis-identified the lanes in several frames which could potentially lead to dangerous driving actions. In the 2nd attempt, we trialed several approaches, detailed in the Jupyter notebook [pipeline-video-alternative.ipynb](./pipeline-video-alternative.ipynb).

(1) - Plotting the intermediate lane detection step side-by-side with the final detection result to help debugging. This visualization reveals that the lane detection pipeline failed where there are trees casting heavy shades onto the road, making a large patch of pixels in the binary image.

(2) - Take derivative in the S channel instead:

```python
sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
```
This make the left lane (white color) a bit more unstable. Thus we did not adopt this approach.


(3) - Sanity check to see if the lanes detected in the current frame deviates significantly from the previously detected lanes (by at most 0.5m).

```python
def sane_lane(left_fitx, right_fitx, left_fit, right_fit, THRESHOLD=700/3.7/2):
    global sane_left
    global sane_right
    global PREV_left_fit
    global PREV_right_fit
    
    violation = False
    if np.average(np.abs(left_fitx - sane_left)) < THRESHOLD:
        sane_left = left_fitx
        PREV_left_fit = left_fit
    else: 
        violation = True
        #lane deviates too much from the previous frame, keep previous detection   
        left_fit = PREV_left_fit
    if np.average(np.abs(right_fitx - sane_right)) < THRESHOLD:
        sane_right = right_fitx
        PREV_right_fit = right_fit
    else: 
        violation = True
        #lane deviates too much from the previous frame, keep previous detection   
        right_fit = PREV_right_fit
    return violation, left_fit, right_fit
```

(4) - Smoothing out curve fitting, by merging the lane pixels detected in the previous frame with the current frame. 

In the end, we found that combining (1) and (4) works well in improving the lane detection results, presented in [project_video_proccessed_2nd.mp4](./project_video_proccessed_2nd.mp4))








