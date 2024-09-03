import numpy as np
import cv2 
import math
def grid_images(images,cols,rescale = 400):
    rows = math.ceil(len(images)/cols)
    combined_img = np.zeros((rescale*rows,rescale*cols,3),dtype=np.uint8)
    for idx,img in enumerate(images):
        row = idx//cols
        col = idx%cols
        if len(img.shape)== 2: img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img,(rescale,rescale))
        # print(combined_img[rescale*row:rescale*(row+1),rescale*col:rescale*(col+1),:].shape)
        combined_img[rescale*row:rescale*(row+1),rescale*col:rescale*(col+1),:] = img
    return combined_img


def reducing_point(contours,resolution = 0.01):
    for idx,c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        contours[idx] = cv2.approxPolyDP(c,resolution * peri,1)
    return contours

def get_four_point(contours):
    length = len(contours)
    idx = 0
    pointer = 0
    while idx<length:
        if len(contours[pointer])!=4:
            contours.pop(pointer)
        else:
            pointer+=1
        idx+=1        
    return contours
        
def get_area(contours,idx):
    return contours[idx]



def modify_four_points(contours):
    # contours = np.array(contours)
    sum_coord = contours.sum(-1)
    minus_coord = contours[:,:,0]-contours[:,:,1]
    first_point = sum_coord.argmin()
    second_point = minus_coord.argmax()
    third_point = minus_coord.argmin()
    fourth_point = sum_coord.argmax()

    return contours[[first_point,second_point,third_point,fourth_point],:,:].reshape(4,-1).tolist()

def draw_contours(img,contours,color = (0,0,255),is_show_length = True,is_show_idx = True,point_mode = False):
    if point_mode == False:
        cv2.drawContours(img, contours, -1, color, 3)
    for idx,c in enumerate(contours):
        if point_mode == True:
            cv2.drawContours(img, contours[idx], -1,color, 10)
        if is_show_idx:
            cv2.putText(img,f"{idx}",contours[idx][0][0],1,1,(255,0,0),2)
        if is_show_length:
            cv2.putText(img,f"{len(c)}",contours[idx][0][0],1,1,(255,0,0),2)
    
def split_img(img,row_points,columns_points):
    rows = np.vsplit(img,row_points,)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,columns_points)
        boxes.append(cols)
    return boxes

def computing_score(acutal_score:np.ndarray,answer:np.ndarray):
    return np.mean(np.equal(acutal_score,answer))*100
    

def computing_cx_cy_radius(img):
    cx_avg = []
    cy_avg = []
    radius_avg = []
    matrix = np.array(split_img(img,5,5))
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
                cx_avg.append(x_axis)
                cy_avg.append(y_axis)
                radius_avg.append(radius)
    return np.mean(cx_avg).astype(np.int32),np.mean(cy_avg).astype(np.int32),np.mean(radius_avg).astype(np.int32)


# actual_ans= np.array([1,2,0,2,4])

def plotting_result(img,cx,cy,radius,actuall_ans:np.ndarray,ans:np.ndarray):
    matrix = np.array(split_img(img,5,5))
    new_matrix= []
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if col == actuall_ans[row] and col == ans[row]:
                cv2.circle(matrix[row][col],(cx,cy),radius,(0,255,0),10)
            elif col == actuall_ans[row] and col != ans[row]:
                cv2.circle(matrix[row][col],(cx,cy),radius,(0,255,0),10)
            elif col != actuall_ans[row] and col == ans[row]:
                cv2.circle(matrix[row][col],(cx,cy),radius,(0,0,255),10)
        new_matrix.append(np.hstack(matrix[row]))
    new_matrix = np.array(new_matrix)
    new_matrix = np.vstack(new_matrix)
    return new_matrix
