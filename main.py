import cv2
import numpy as np
import math
from utils import (draw_contours,
                   grid_images,
                   reducing_point,
                   get_four_point,
                   get_area,
                   modify_four_points,
                   split_img,
                   computing_score,
                   computing_cx_cy_radius,plotting_result
)
HEIGHT_IMG = 800
WIDHT_IMG  = 800

HEIGHT_GRADE_AREA = 150
WIDTH_GRADE_AREA = 325

actual_ans= np.array([1,2,0,2,4])


img = cv2.imread("./Data/3.jpg")
img = cv2.resize(img,(HEIGHT_IMG,WIDHT_IMG))

grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(grey_img,(5,5),1)
canny_images = cv2.Canny(img_gaussian,10,70)


# manipulate contour
contours, hierarchy = cv2.findContours(canny_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imgContours = img.copy()
draw_contours(imgContours,contours,(0, 75, 150),is_show_idx=False)
# 
imgContours_clone = img.copy()
contours_clone = list(contours).copy()
reducing_point(contours_clone,0.01)
get_four_point(contours_clone)
draw_contours(imgContours_clone,contours_clone,(0, 75, 150),is_show_length=False,point_mode=True)


cv2.imshow("contours",imgContours_clone)


# get areas

results_contours = get_area(contours_clone,2)
score_contours = get_area(contours_clone,1)

# answer area 
results_contours=modify_four_points(results_contours)
pts1 = np.float32(results_contours)
pts2 = np.float32([[0, 0],[WIDHT_IMG, 0], [0, HEIGHT_IMG],[WIDHT_IMG, HEIGHT_IMG]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
answer_imgWarpColored = cv2.warpPerspective(img, matrix, (WIDHT_IMG, HEIGHT_IMG))

# score area
score_contours=modify_four_points(score_contours)
pts1_score = np.float32(score_contours)
pts2_score = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
matrix = cv2.getPerspectiveTransform(pts1_score, pts2_score)
score_imgWarpColored = cv2.warpPerspective(img, matrix, (WIDTH_GRADE_AREA, HEIGHT_GRADE_AREA))




# applying gray,threshold
choices_imgWarpColored = cv2.cvtColor(answer_imgWarpColored,cv2.COLOR_BGR2GRAY)
choices_imgWarpColored = cv2.threshold(choices_imgWarpColored, 170, 255, cv2.THRESH_BINARY_INV)[1]

matrix = np.array(split_img(choices_imgWarpColored,5,5))

# counting non zero pixel and finding out the maximum index
funct = lambda x: np.argmax(np.array([[np.count_nonzero(x[i][y]) for y in range(x.shape[1])] for i in range(x.shape[0])]),1)
ans = funct(matrix)

# COMPARE THE VALUES TO FIND THE CORRECT ANSWERS
grading = computing_score(ans,actual_ans)

# taking cx,cy,radius
cx,cy,radius = computing_cx_cy_radius(answer_imgWarpColored)

# plotting result
new_matrix = plotting_result(answer_imgWarpColored,cx,cy,radius,actual_ans,ans)
# Display result 
imgRawscore = np.zeros_like(answer_imgWarpColored,np.uint8) 
imgRawscore = plotting_result(imgRawscore,cx,cy,radius,actual_ans,ans)
invMatrixG = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
imgInvResultDisplay = cv2.warpPerspective(imgRawscore, invMatrixG, (WIDHT_IMG, HEIGHT_IMG))



# Display grade
imgRawGrade = np.zeros_like(score_imgWarpColored,np.uint8) 
cv2.putText(imgRawGrade,str(int(grading))+"%",(70,100)
                        ,cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),3)
invMatrixG = cv2.getPerspectiveTransform(pts2_score, pts1_score) # INVERSE TRANSFORMATION MATRIX
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (WIDHT_IMG, HEIGHT_IMG))



imgFinal = img.copy()
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvResultDisplay, 1,0)
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1,0)
# imgFinal = cv2.bitwise_or(imgFinal,imgInvGradeDisplay)


# Showing image case
combine_images = grid_images([img,grey_img,img_gaussian,canny_images,imgContours,
                              imgContours_clone,answer_imgWarpColored,
                              choices_imgWarpColored,new_matrix,imgInvResultDisplay,imgFinal],6,700)
cv2.namedWindow('combined_img', cv2.WINDOW_NORMAL)
cv2.imshow("combined_img",combine_images)
# cv2.imshow("split_image",matrix[0][1])
# cv2.imshow("score_area",score_imgWarpColored)
# cv2.imshow("score",imgInvGradeDisplay)
cv2.imwrite("./img/final_result.jpg",combine_images)
cv2.waitKey(0)

