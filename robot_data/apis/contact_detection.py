import cv2
import numpy as np

def make_kernal(n,type):
    if(type=='circle'):
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    elif(type=='rect'):
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT,(n,n))
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_CROSS,(n,n))
    return kernal 

kernal = make_kernal(11,'circle')
kernal1 = make_kernal(6,'circle')
kernal1_1 = make_kernal(5,'circle')
kernal2 = make_kernal(11,'rect')
kernal3 = make_kernal(31,'circle')
kernal4 = make_kernal(5,'circle')
kernal5 = make_kernal(5,'rect')
kernal6 = make_kernal(25,'circle')
kernal_size = 25
kernal7 = make_kernal(kernal_size,'circle')
kernal8 = make_kernal(2,'rect')
kernal9 = make_kernal(2,'rect')
kernal10 = make_kernal(35,'circle') #25
kernal11 = make_kernal(4,'circle')

cols, rows, cha = 240, 320, 3 
def contact_detection(im_cal):
    pad = 35  # 20
    highbar_top = 50 #120
    lowbar_top = 27   #90
    highbar_down = 50 #70
    lowbar_down = 27  #50
    im_canny_top = cv2.Canny(im_cal[:int(rows*2/3),:].astype(np.uint8), lowbar_top, highbar_top, 11)
    im_canny_down = cv2.Canny(im_cal[int(rows*2/3):,:].astype(np.uint8), lowbar_down, highbar_down, 11)

    im_canny = np.concatenate((im_canny_top,im_canny_down),axis = 0)

    im_canny[:,:pad] = 0
    im_canny[:,-pad:] = 0
    im_canny[:20,:] = 0
    im_canny[-20:,:] = 0
    # cv2.imshow('canny_image',im_canny) 
    # im_canny[-pad:,:] = 0

    # im_canny = im_canny  * de_mask # used


    # im_canny = mask_blue * im_canny  * mask_brightness
    # cv2.imshow('calibrated image', im_cal.astype(np.uint8))

    # cv2.imshow('canny_imag_without_mask',im_canny) 
    # cv2.waitKey(1)
    img_d = cv2.dilate(im_canny, kernal1, iterations=1)
    # cv2.imshow('img_d',img_d)
    img_dd = cv2.medianBlur(img_d,7)
    # img_dd = cv2.dilate(img_d, kernal11, iterations=1)
    # cv2.imshow('img_dd',img_dd)
    img_e = cv2.erode(img_dd, kernal1_1, iterations=1)
    # cv2.imshow('img_e',img_e)
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    # cv2.imshow('img_ee',img_ee)
    contact = cv2.dilate(img_ee, kernal3, iterations=1).astype(np.uint8)
    # cv2.imshow('contact',contact)
    # cv2.waitKey(0)
    return contact

def _smooth(target):
    kernel = np.ones((27, 27), np.float32)#64
    kernel /= kernel.sum()
    diff_blur = cv2.filter2D(target, -1, kernel)
    return diff_blur

def contact_detection_v2(frame, origin):
    diff = cv2.absdiff(frame, origin)
    # diff = _diff(frame, origin)
    # cv2.imshow('diff',diff)
    mask = _smooth(diff)
    b,g,r=cv2.split(mask)
    gray_img=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    # 尝试用阈值限定但是泛化性很差
    #############################################################
    _, maskn = cv2.threshold(r, 24, 255, cv2.THRESH_BINARY)
    # _, maskn = cv2.threshold(gray_img, 36, 255, cv2.THRESH_BINARY)
    # _, maskn = cv2.threshold(g, 34, 255, cv2.THRESH_BINARY)
    maskn = cv2.medianBlur(maskn,15)
    # cv2.imshow('maskn',maskn)
    #############################################################
    # cv2.imshow('b',r)
    im_canny_top1 = cv2.Canny(r, 6, 7, 5)
    # im_canny_top2 = cv2.Canny(gray_img, 6, 7, 9)
    # cv2.imshow('im_canny_top',im_canny_top1)
    img_d = cv2.dilate(im_canny_top1, kernal1, iterations=1)
    img_dd = cv2.dilate(img_d, kernal11, iterations=1)
    img_e = cv2.erode(img_dd, kernal1_1, iterations=1)
    img_ee = cv2.erode(img_e, kernal2, iterations=1)
    contact = cv2.dilate(img_ee, kernal3, iterations=1).astype(np.uint8)
    # contact = cv2.medianBlur(contact,25)
    return contact