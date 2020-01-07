import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# The pipeline function takes in a numpy array of dimensions:
#   "height, width, color-space" 
# and MUST return an image of the SAME dimensions
#
# The pipeline function also takes a motorq. To make the motors move
# add messages to the queue of the form:
#   motorq.put( [ left-motor-speed , right-motor-speed ] )
# i.e.  motorq.put([32768,32768]) # make the motors go full-speed forward
def pipeline(image,motorq):
	print("running pipeline...")

	# THINGS YOU SHOULD DO...
	# 1. Copy the code INSIDE your pipeline function here.
	# 2. Ensure the pipeline function takes BOTH the image and motorq.

	#motorq.put([32768,32768]) # make the motors go full-speed forward
        height = image.shape[0]
        width = image.shape[1]
	
	
	
	
        """region_of_interest_vertices = [
                (0, height*2 / 3),
		(0, height),
		(width, height),
		(width, height*2/3),
                (width / 2, height / 2),
        ]"""
        """region_of_interest_vertices = [
		(0, height),
		(width, height),
                (width / 2, height / 2)
        ]"""
        region_of_interest_vertices = [
		(0, height/3),
		(0, height/3),
		(width, height/3),
		(width, height/3)
	]

        kernel = np.ones((4,4),np.float32)/25
        blurred_image = cv2.filter2D(image,-1,kernel)

        #unwarped_image = unwarp_img(blurred_image)
	
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)

        black_white_img = blackwhite(gray_image)
	
        cannyed_image = cv2.Canny(gray_image, 100, 200)
 
        cropped_image = region_of_interest(
                cannyed_image,
                np.array(
                        [region_of_interest_vertices],
                        np.int32
                ),
        )

        
	
	
 
        lines = cv2.HoughLinesP(
                cropped_image,
                rho=6,
                theta=np.pi / 60,
                threshold=160,
                lines=np.array([]),
                minLineLength=40,
                maxLineGap=25
        )
 
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
    
        if not np.any(lines):
                return image
        
        for line in lines:
                for x1, y1, x2, y2 in line:
			if(x2 != x1):
				slope = float(y2 - y1) / (x2 - x1)
				if math.fabs(slope) < 0.5:
					continue
				if slope <= 0:
					left_line_x.extend([x1, x2])
					left_line_y.extend([y1, y2])
				else:
					right_line_x.extend([x1, x2])
					right_line_y.extend([y1, y2])

        if len(left_line_x)==0 or len(right_line_x)==0:
                return image

        min_y = int(image.shape[0] * (3 / 5))
        max_y = int(image.shape[0])

        poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
        ))
 
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
 
        poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
        ))
 
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        """line_image = draw_lines(
                image,
                [[
                        [left_x_start, max_y, left_x_end, min_y],
                        [right_x_start, max_y, right_x_end, min_y],
                ]],
                thickness=5,
        )"""

        return black_white_img

def blackwhite(image):
        for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                        #h, s, v = rgb_to_hsv(image[i][j][0], image[i][j][1], image[i][j][2])
                        if(image[i][j] <= 130):
                                image[i][j] = 0
                        else:
                                image[i][j] = 255

        return image

        
def region_of_interest(img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

def draw_lines(img, lines, color=[0, 255, 255], thickness=3):
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        img = np.copy(img)
        if lines is None:
                return

        for line in lines:
                for x1, y1, x2, y2 in line:
                        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

        img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

        return img

def unwarp_img(image):
        K = np.array([[  404.41/1.3,     0.  ,  486/2],
              [    0.  ,   302.89/1.3,   364/2],
              [    0.  ,     0.  ,     1.  ]])
        # zero distortion coefficients work well for this image
        D = np.array([0., 0., 0., 0.])

        # use Knew to scale the output
        Knew = K.copy()
        Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]

        temp_image = cv2.fisheye.undistortImage(image, K, D = D, Knew = Knew)
        undistorted_image = temp_image[(h/3-30):(h/3-50)+h/2, w/4:w/4+w/2]

        return undistorted_image
        
"""def pipeline(image):
    
    An image processing pipeline which will output
    an image with the lane lines annotated.
    

    height = image.shape[0]
    width = image.shape[1]
	
	
	
	
    region_of_interest_vertices = [
        (0, height*2 / 3),
		(0, height),
		(width, height),
		(width, height*2/3),
        (width / 2, height / 2),
    ]
    region_of_interest_vertices = [
		(0, height),
		(width, height),
        (width / 2, height / 2)
    ]
    region_of_interest_vertices = [
		(0, height/2),
		(0, height),
		(width, height),
		(width, height/2)
	]

    kernel = np.ones((4,4),np.float32)/25
    blurred_image = cv2.filter2D(image,-1,kernel)
	
    gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2GRAY)
	
    cannyed_image = cv2.Canny(gray_image, 100, 200)
 
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
	
	
 
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )
 
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    
    if not np.any(lines):
        return image
        
    for line in lines:
        for x1, y1, x2, y2 in line:
			if(x2 != x1):
				slope = float(y2 - y1) / (x2 - x1)
				if math.fabs(slope) < 0.5:
					continue
				if slope <= 0:
					left_line_x.extend([x1, x2])
					left_line_y.extend([y1, y2])
				else:
					right_line_x.extend([x1, x2])
					right_line_y.extend([y1, y2])

    if len(left_line_x)==0 or len(right_line_x)==0:
        return image

    min_y = int(image.shape[0] * (3 / 5))
    max_y = int(image.shape[0])

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
 
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
 
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
 
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )

    return line_image"""
