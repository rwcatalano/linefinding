#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_long_lines(img, lines, color=[0, 255, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    #append all x's to this array then sort array to get max and min...
    #Left line
    left_x = []
    left_x2 = []
    left_y = []
    left_y2 = []
    leftSlope = []
    
    #Right line
    right_x = []
    right_x2 = []
    right_y = []
    right_y2 = []
    rightSlope = []
    
    #you can figure out tops and bottoms from your region of interest
    y_min = img.shape[0] #run to the middle of the image
    y_max = img.shape[0]/2+40 #run to the bottom of the image
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            d = (x2-x1)
            
            if(d==0):
                d=0.0000001
            
            slope = ((y2-y1)/d)

            if(slope<0):
                left_x.append(x1)
                left_x2.append(x2)
                left_y.append(y1)
                left_y2.append(y2)
                leftSlope.append(slope)
            else:
                print(x1,y1)
                right_x.append(x1)
                right_x2.append(x2)
                right_y.append(y1)
                right_y2.append(y2)
                rightSlope.append(slope)
        
    
    left_slope_mean = np.mean(leftSlope)
    if(left_slope_mean == 0):
        left_slope_mean == 0.001
    
    left_x1_mean = (np.mean(left_x))
    left_y1_mean = (np.mean(left_y))
    
    left_x2_mean = (np.mean(left_x2))
    left_y2_mean = (np.mean(left_y2))
    
    left_yint_1 = left_y2_mean - (left_slope_mean)*left_x2_mean
    left_yint_2 = left_y1_mean - (left_slope_mean)*left_x1_mean
    
    left_xint_1 = (left_yint_1 * -1)/left_slope_mean
    left_xint_2 = (left_yint_2 * -1)/left_slope_mean
    
    """
    print('MEANS......')
    print('Left Slope:', left_slope_mean)  
    print('---------------')
    print('first pair')
    print('left x', left_x1_mean)
    print('left y', left_y1_mean)
    print('---------------')
    print('second pair')
    print('left x2', left_x2_mean)
    print('left y2', left_y2_mean)
    print('---------------')
    print('---------------')
    
    print('INTERCEPTS')
    print('y-int-1:', left_yint_1)
    print('y-int-2:', left_yint_2)
    
    print('INTERCEPTS')
    print('x-int-1:', left_xint_1)
    print('x-int-2:', left_xint_2)
    """
    
    right_slope_mean = np.mean(rightSlope)
    if(right_slope_mean == 0):
        right_slope_mean == 0.001
    
    right_x1_mean = (np.mean(right_x))
    right_y1_mean = (np.mean(right_y))
    
    right_x2_mean = (np.mean(right_x2))
    right_y2_mean = (np.mean(right_y2))
    
    right_yint_1 = right_y2_mean - (right_slope_mean)*right_x2_mean
    right_yint_2 = right_y1_mean - (right_slope_mean)*right_x1_mean 
    
    right_xint_1 = (right_yint_1 * -1)/right_slope_mean    
    right_xint_2 = (right_yint_2 * -1)/right_slope_mean
    
    """
    print('MEANS......')
    print('Right Slope:', right_slope_mean)  
    print('---------------')
    print('first pair')
    print('left x', right_x1_mean)
    print('left y', right_y1_mean)
    print('---------------')
    print('second pair')
    print('left x2', right_x2_mean)
    print('left y2', right_y2_mean)
    print('---------------')
    print('---------------')
    
    print('INTERCEPTS')
    print('y-int-1:', right_yint_1)
    print('y-int-2:', right_yint_2)
    
    print('INTERCEPTS')
    print('x-int-1:', right_xint_1)
    print('x-int-2:', right_xint_2)
    """
    
    leftx1 = int((y_min-left_yint_1)/left_slope_mean)
    lefty1 = int(y_min)
    leftx2 = int((y_max-left_yint_2)/left_slope_mean)
    lefty2 = int(y_max)
    
    
    
    if(np.isnan(right_yint_1)):
        right_yint_1 = 0
        
    if(np.isnan(right_yint_2)):
        right_yint_2 = 0
    
    if(np.isnan(right_slope_mean)):
        right_slope_mean = 0.000001;
    
    rx1pos = int(y_min-right_yint_1)
    rx1pos = int(y_min-right_yint_1)
    
    rightx1 = int(rx1pos/right_slope_mean)
    righty1 = int(y_min)
    rightx2 = int((y_max-right_yint_2)/right_slope_mean)
    righty2 = int(y_max)
    
    #line is in the right spot... wtf I cannot seem to get the yint and xint...
    #cv2.line(img, (int(right_x1_mean),int(right_y1_mean)), (int(right_x2_mean),int(right_y2_mean)), color, thickness)
    cv2.line(img, (leftx1, lefty1), (leftx2, lefty2), color, thickness)
    cv2.line(img, (rightx1, righty1), (rightx2, righty2), color, thickness)
    
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_long_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
	
	
	
	
	
	
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)

    #drop color
    gray  = grayscale(image)

    #soften the image
    blurGray = gaussian_blur(gray,7)

    #canny edge detection
    cannyImg = canny(blurGray, 20, 50)
    
    
    ysize = image.shape[0]
    xsize = image.shape[1]
    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [xsize*.5, ysize*.6]
    verts = np.array([left_bottom,right_bottom,apex])

    #rwc refactor - draw a trap vertices to get region of interest
    fac_imgY = 1.65
    x_max = np.shape(image)[1] #height
    y_max = np.shape(image)[0] #width
    vertices = np.array([[(0,y_max),(x_max,y_max), (x_max/2+30,y_max/fac_imgY),
                        (x_max/2-30,y_max/fac_imgY)]], dtype = np.int32)


    regImg = region_of_interest(cannyImg, vertices)
    
    #variables for line detection
    rho = 1
    theta= np.pi/180
    threshold=45
    min_line_len=20
    max_line_gap = 5

    #performs line detection
    houghImg = hough_lines(regImg, rho, theta, threshold, min_line_len, max_line_gap)
    
    weightedImg = weighted_img(houghImg, image, α=0.8, β=1., λ=0.)
    plt.imshow(weightedImg)
    
    return weightedImg
	
	
	
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
%time white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
%time yellow_clip.write_videofile(yellow_output, audio=False)