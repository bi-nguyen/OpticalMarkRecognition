import cv2
from utils import draw_contours,split_img,get_area,reducing_point,computing_cx_cy_area
import numpy as np
img = cv2.imread("result.jpg")

print(img.shape)
matrix = np.array(split_img(img,5,5))

print(matrix.shape)

new_matrix = []
areas = []
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        img1 = matrix[row][col]
        grey_img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img_gaussian = cv2.GaussianBlur(grey_img,(5,5),1)
        canny_images = cv2.Canny(img_gaussian,50,70)
        contours, hierarchy = cv2.findContours(canny_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==1:
            # draw_contours(img1,contours,(0, 75, 150),is_show_idx=True,is_show_length=False)
            (x_axis,y_axis),radius = cv2.minEnclosingCircle(contours[0]) 
            print(x_axis,y_axis,radius)
            # area = cv2.contourArea(contours[0])
            # radius = np.sqrt(area / np.pi) 
            # areas.append(radius)
            # print(radius)
            cv2.circle(img1,(int(x_axis),int(y_axis)),int(radius),(0,255,0),2) 
            cv2.imshow("contour",img1)
            cv2.waitKey(0)
    new_matrix.append(np.hstack(matrix[row]))
    
        
grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(grey_img,(5,5),1)
canny_images = cv2.Canny(img_gaussian,50,70)

new_matrix = np.array(new_matrix)
new_matrix = np.vstack(new_matrix)
print(areas)

# grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# img_gaussian = cv2.GaussianBlur(grey_img,(5,5),1)
# canny_images = cv2.Canny(img_gaussian,10,70)
# contours, hierarchy = cv2.findContours(canny_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# imgContours = img.copy()
# draw_contours(imgContours,contours,(0, 75, 150),is_show_idx=True,is_show_length=False)






# contours_clone = list(contours).copy()
# results_contours = get_area(contours_clone,0)
# imgContours = img.copy()
# cv2.drawContours(imgContours, contours, -1, color=(0, 0, 255), thickness=cv2.FILLED)



# new_image = np.hstack([img,new_matrix])



# cv2.imshow("canny_images",new_image)
# cv2.waitKey(0)
