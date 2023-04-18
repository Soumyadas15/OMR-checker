import cv2
import os
import pandas as pd
import numpy as np
import four_point

# define the answer key as a hashmap
answer_key = {1 : 1,
              2 : 1,
              3 : 1,
              4 : 1,
              5 : 1,
              6 : 1,
              7 : 1,
              8 : 1,
              9 : 1,
              10 : 1,
              11 : 1,
              12 : 1,
              13 : 1,
              14 : 1,
              15 : 1,
              16 : 1,
              17: 1,
              18 : 1,
              19 : 1,
              20 : 1,
              21 : 1,
              22 : 1,
              23 : 1,
              24 : 1,
              25 : 1,
              26 : 1,
              27 : 1,
              28 : 1,
              29 : 1,
              30 : 1}

# create a dictionary to store scores
scores = {}
questions = 30
choices = 4

# loop over all images in the folder
for image_file in os.listdir("images"):
    # read the image
    image = cv2.imread(os.path.join("images", image_file))
    
    if(image is not None):
    
        # convert to grayscale and apply thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # find contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # create a mask for the largest contour
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        src_corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

        # Define the size of the output image
        output_size = (w, h)

        # Define the corners of the rectangle in the output image
        dst_corners = np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)

        # Apply the perspective transform
        warped = cv2.warpPerspective(image, M, output_size)

        # Display the warped image
        # cv2.imshow('Warped', warped)
        # cv2.waitKey(0)
        
        # apply the mask to the thresholded image
        largest_img_contours = cv2.drawContours(image, [largest_contour], 0, (0, 255, 0), 2)
        masked = cv2.bitwise_and(thresh, thresh, mask=mask)
        # cv2.imshow('Largest contour', img_contours)
        # cv2.waitKey(0)
        
        #Now define threshoold for marking points
        
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image
        warped_gray_inv = cv2.bitwise_not(warped_gray)

        # Threshold the inverted grayscale image
        _, warped_thresh = cv2.threshold(warped_gray_inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow('Warped', warped_thresh)
        # cv2.waitKey(0)
        
        #Computing individual bubbles
        num_rows = questions
        num_cols = choices

        # get the dimensions of the image
        height, width = warped_thresh.shape

        # calculate the height and width of each box
        box_height = height // num_rows
        box_width = width // num_cols

        # calculate the coordinates of each box
        box_coords = []
        for i in range(num_rows):
            for j in range(num_cols):
                top = i * box_height
                bottom = (i + 1) * box_height
                left = j * box_width
                right = (j + 1) * box_width
                box_coords.append((top, bottom, left, right))

        # split the marked image into boxes
        boxes = []
        for coords in box_coords:
            top, bottom, left, right = coords
            box = warped_thresh[top:bottom, left:right]
            boxes.append(box)

        # display the first row of boxes
        second_row = boxes[num_cols:num_cols*2]
        box_to_display = second_row[0]
        # cv2.imshow('First Element of Second Row', boxes[7])
        # cv2.waitKey(0)
        # cv2.waitKey(0)
        
        
        
        # initialize the list of marked answers
        counts = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                box = boxes[i*num_cols + j]
                count = np.count_nonzero(box)
                counts[i, j] = count    

        #Retrieving answers
        marked_answer = [0]
        incorrect = []
        
        for i in range(num_rows):
            min_count = np.min(counts[i])
            marked = []
            for j in range(num_cols):
                count = counts[i,j]
                if count > min_count * 1.8:
                    marked.append(j)
            if len(marked) == 1:
                marked_answer.append(marked[0] + 1)
            elif len(marked) > 1 and len(marked) == 0:
                marked_answer.append('N')
            else:
                marked_answer.append('N')
                incorrect.append(i)
   
        
        # compare the marked answers with the answer key
        score = 0
        plus = []
        for i in range(1,31):
            if marked_answer[i] == answer_key[i]:
                score += 1
        
        # # store the score in the dictionary
        scores[image_file] = score
        
        

# save the scores in a CSV file
df = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
df.index.name = 'image'
df.to_csv('scores.csv')
print("Results with qualified teams acheived successfully")
