import cv2
import numpy as np
#import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy import signal
from tqdm import tqdm
import time

class marker_detector:
    def __init__(self):

        # 需要修改###########################################################
        self.BallRad = 4.5 / 2  # 用于标定的球半径, mm
        self.Pixmm = 20.75 / 240.0  # 每一个像素宽度对应的长度, mm/pixel
        # 需要修改###########################################################

        self.ratio = 1 / 2.
        self.red_range = [-90, 90]
        self.green_range = [-90, 90]
        self.blue_range = [-90, 90]
        self.red_bin = int((self.red_range[1] - self.red_range[0]) * self.ratio)
        self.green_bin = int((self.green_range[1] - self.green_range[0]) * self.ratio)
        self.blue_bin = int((self.blue_range[1] - self.blue_range[0]) * self.ratio)
        self.zeropoint = [-90, -90, -90]
        self.lookscale = [180., 180., 180.]
        self.bin_num = 90

    # old
    def mask_marker(self, raw_image):
        """"提取marker，输出二值白底黑marker图"""

        m, n = raw_image.shape[1], raw_image.shape[0]

        raw_image = cv2.pyrDown(raw_image).astype(np.float32)

        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        mask_b = diff[:, :, 0] > 150
        mask_g = diff[:, :, 1] > 150
        mask_r = diff[:, :, 2] > 150
        mask = (mask_b * mask_g + mask_b * mask_r + mask_g * mask_r) > 0
        mask = cv2.resize(mask.astype(np.uint8), (m, n))

        return (1 - mask) * 255
    
    # new
    def make_mask(self, raw_image):
        """"提取marker，输出二值白底黑marker图"""
        gray = raw_image[:,:,1] ### use only the green channel
        im_blur_3 = cv2.GaussianBlur(gray,(3,3),5)
        im_blur_8 = cv2.GaussianBlur(gray, (15,15),5)
        im_blur_sub = im_blur_8 - im_blur_3 + 128
        mask = cv2.inRange(im_blur_sub, 140, 255)
        mask[:,:45] = 0
        mask[:,-45:] = 0

        # ''' normalized cross correlation '''
        template = gkern(l=20, sig=3)
        nrmcrimg = normxcorr2(template, mask)
        # ''''''''''''''''''''''''''''''''''''
        a = nrmcrimg
        mask = np.asarray(a > 0.1)
        mask = (mask).astype('uint8')

        return (1 - mask) * 255
    
    # new
    def marker_center(self, mask):
        ''' second method '''
        img3 = mask
        neighborhood_size = 10
        threshold = 0 # for mini
        data_max = maximum_filter(img3, neighborhood_size)
        maxima = (img3 == data_max)
        data_min = minimum_filter(img3, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        MarkerCenter = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
        MarkerCenter[:, [0, 1]] = MarkerCenter[:, [1, 0]]
        # for i in range(MarkerCenter.shape[0]):
        #     x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
        #     cv2.circle(mask, (x0, y0), color=(0, 0, 255), radius=1, thickness=1)
        return MarkerCenter

    # old
    def find_dots(self, binary_image):
        """造一个SimpleBlobDetector，返回marker的坐标（应该是中心，待考证）"""

        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))

        return keypoints

    # new
    def vis_mask(self, mask, MarkerCenter):
        """画marker掩膜，给每个marker上画椭圆或圆，圆内填充1"""
        mask = np.zeros_like(mask[:, :, 0])
        for i in range(MarkerCenter.shape[0]):
            x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
            cv2.circle(mask, (x0, y0), color=(0, 0, 255), radius=9, thickness=1)
        return mask

    def contact_detection(self, raw_image, ref, marker_mask, idx):
        """用于手动标定每一张标定图片中球的位置和大小、
           w/s/a/d用于控制黑色圆圈的上下左右，m/n控制大小
           调整完毕后按Esc进入下一张"""
        
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0)
        diff_img = np.max(np.abs(raw_image.astype(np.float32) - blur), axis=2)
        contact_mask = (diff_img > 100).astype(np.uint8) * (1 - marker_mask)

        # auto find init point
        # contours, hierarchy = cv2.findContours(contact_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # areas = [cv2.contourArea(c) for c in contours]
        # sorted_areas = np.sort(areas)
        # cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x), int(y))
        # radius = int(radius)

        # manual set init point 
        (x, y) = (raw_image.shape[0] / 2, raw_image.shape[1] / 2)
        center = (int(x), int(y))
        radius = int(45)

        key = -1
        while key != 27:
            center = (int(x), int(y))
            radius = int(radius)
            im2show = cv2.circle(np.array(raw_image), center, radius, (0, 40, 0), 1)
            cv2.imshow(idx, im2show.astype(np.uint8))
            key = cv2.waitKey(0)
            if key == 119:
                y -= 1
            elif key == 115:
                y += 1
            elif key == 97:
                x -= 1
            elif key == 100:
                x += 1
            elif key == 109:
                radius += 1
            elif key == 110:
                radius -= 1

        contact_mask = np.zeros_like(contact_mask)
        cv2.circle(contact_mask, center, radius, (1), -1)
        contact_mask = contact_mask * (1 - marker_mask)

        return contact_mask, center, radius

    def get_gradient_v2(self, img, ref, center, radius_p, valid_mask, table, table_account):
        """计算梯度，存入table"""

        ball_radius_p = self.BallRad / self.Pixmm
        blur = cv2.GaussianBlur(ref.astype(np.float32), (3, 3), 0) + 1
        blur_inverse = 1 + ((np.mean(blur) / blur) - 1) * 2
        img_smooth = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
        diff_temp1 = img_smooth - blur
        diff_temp2 = diff_temp1 * blur_inverse

        diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] - self.zeropoint[0]) / self.lookscale[0]
        diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] - self.zeropoint[1]) / self.lookscale[1]
        diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] - self.zeropoint[2]) / self.lookscale[2]
        diff_temp3 = np.clip(diff_temp2, 0, 0.999)
        diff = (diff_temp3 * self.bin_num).astype(int)
        pixels_valid = diff[valid_mask > 0]

        x = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y = np.linspace(0, img.shape[1] - 1, img.shape[1])
        xv, yv = np.meshgrid(y, x)
        xv = xv - center[0]
        yv = yv - center[1]
        rv = np.sqrt(xv ** 2 + yv ** 2)

        radius_p = min(radius_p, ball_radius_p - 1)
        mask = (rv < radius_p)
        mask_small = (rv < radius_p - 1)

        temp = ((xv * mask) ** 2 + (yv * mask) ** 2) * self.Pixmm ** 2
        height_map = (np.sqrt(self.BallRad ** 2 - temp) * mask - np.sqrt(
            self.BallRad ** 2 - (radius_p * self.Pixmm) ** 2)) * mask
        height_map[np.isnan(height_map)] = 0

        gx_num = signal.convolve2d(height_map, np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]), boundary='symm',
                                   mode='same') * mask_small
        gy_num = signal.convolve2d(height_map, np.array([[0, 0, 0], [0.5, 0, -0.5], [0, 0, 0]]).T, boundary='symm',
                                   mode='same') * mask_small
        gradxseq = gx_num[valid_mask > 0]
        gradyseq = gy_num[valid_mask > 0]

        for i in range(gradxseq.shape[0]):
            b, g, r = pixels_valid[i, 0], pixels_valid[i, 1], pixels_valid[i, 2]
            #            print(r,g,b)
            if table_account[b, g, r] < 1.:
                table[b, g, r, 0] = gradxseq[i]
                table[b, g, r, 1] = gradyseq[i]
                table_account[b, g, r] += 1
            else:
                # 平均
                table[b, g, r, 0] = (table[b, g, r, 0] * table_account[b, g, r] + gradxseq[i]) / (
                        table_account[b, g, r] + 1)
                table[b, g, r, 1] = (table[b, g, r, 1] * table_account[b, g, r] + gradyseq[i]) / (
                        table_account[b, g, r] + 1)
                table_account[b, g, r] += 1

        return table, table_account

    def smooth_table(self, table, count_map):
        """对标定的table中没有值的部分进行填充"""

        y, x, z = np.meshgrid(np.linspace(0, self.bin_num - 1, self.bin_num),
                              np.linspace(0, self.bin_num - 1, self.bin_num),
                              np.linspace(0, self.bin_num - 1, self.bin_num))

        unfill_x = x[count_map < 1].astype(int)
        unfill_y = y[count_map < 1].astype(int)
        unfill_z = z[count_map < 1].astype(int)
        fill_x = x[count_map > 0].astype(int)
        fill_y = y[count_map > 0].astype(int)
        fill_z = z[count_map > 0].astype(int)

        fill_gradients = table[fill_x, fill_y, fill_z, :]
        table_new = np.array(table)
        temp_num = unfill_x.shape[0]
        for i in tqdm(range(temp_num)):
            distance = (unfill_x[i] - fill_x) ** 2 + (unfill_y[i] - fill_y) ** 2 + (unfill_z[i] - fill_z) ** 2
            if np.min(distance) < 20:
                index = np.argmin(distance)
                table_new[unfill_x[i], unfill_y[i], unfill_z[i], :] = fill_gradients[index, :]
            # time.sleep(0.02)
        return table_new

def gkern(l=5, sig=1.):
    """ creates gaussian kernel with side length l and a sigma of sig """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)

def normxcorr2(template, image, mode="same"):
    """
    Input arrays should be floating point numbers.
    :param template: N-D array, of template or filter you are using for cross-correlation.
    Must be less or equal dimensions to image.
    Length of each dimension must be less than length of image.
    :param image: N-D array
    :param mode: Options, "full", "valid", "same"
    full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
    Output size will be image size + 1/2 template size in each dimension.
    valid: The output consists only of those elements that do not rely on the zero-padding.
    same: The output is the same size as image, centered with respect to the ‘full’ output.
    :return: N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")
    template = template - np.mean(template)
    image = image - np.mean(image)
    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)
    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))
    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0
    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)
    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    return out


def init(frame):
    RESCALE = 1
    return cv2.resize(frame, (0, 0), fx=1.0/RESCALE, fy=1.0/RESCALE)


def preprocessimg(img):
    '''
    Pre-processing image to remove noise
    '''
    dotspacepx = 36
    ### speckle noise denoising
    # dst = cv2.fastNlMeansDenoising(img_gray, None, 9, 15, 30)
    ### adaptive histogram equalizer
    # clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(10, 10))
    # equalized = clahe.apply(img_gray)
    ### Gaussian blur
    # gsz = 2 * round(3 * mindiameterpx / 2) + 1
    gsz = 2 * round(0.75 * dotspacepx / 2) + 1
    blur = cv2.GaussianBlur(img, (51, 51), gsz / 6)
    #### my linear varying filter
    x = np.linspace(3, 1.5, img.shape[1])
    y = np.linspace(3, 1.5, img.shape[0])
    xx, yy = np.meshgrid(x, y)
    mult = blur * yy
    ### adjust contrast
    res = cv2.convertScaleAbs(blur, alpha=2, beta=0)
    return res


def find_marker(frame):
    # RESCALE = setting.RESCALE
    # # Blur image to remove noise
    # blur = cv2.GaussianBlur(frame, (int(11/RESCALE), int(11/RESCALE)), 0)

    # # subtract the surrounding pixels to magnify difference between markers and background
    # diff = frame.astype(np.float32) - blur

    # diff *= 4.0
    # diff[diff<0.] = 0.
    # diff[diff>255.] = 255.
    # diff = cv2.GaussianBlur(diff, (int(25/RESCALE), int(25/RESCALE)), 0)
    #
    # # # Switch image from BGR colorspace to HSV
    # hsv = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2HSV)
    #
    # # # yellow range in HSV color space
    # yellowMin = (0, 0, 32)
    # yellowMax = (100, 255, 255)
    #
    # # Sets pixels to white if in yellow range, else will be set to black
    # mask = cv2.inRange(hsv, yellowMin, yellowMax)

    #### masking technique for small dots
    # diff = diff.astype('uint8')
    # mask = cv2.inRange(diff, (200, 200, 200), (255, 255, 255))

    # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # gray[gray > 200] = 255;
    # gray[gray < 150] = 0
    # mask = gray

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
    #                            param1=50, param2=12, minRadius=1, maxRadius=20)
    #
    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(diff, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(diff, (i[0], i[1]), 2, (0, 0, 255), 3)


    ##### masking techinique for dots on R1.5
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # res = preprocessimg(gray)
    # # mask = cv2.inRange(gray, 20, 70)
    # # kernel = np.ones((3, 3), np.uint8)
    # # dilation = cv2.dilate(mask, kernel, iterations=1)
    # adjusted = cv2.convertScaleAbs(gray, alpha=3.75, beta=30)
    # mask = cv2.inRange(adjusted, 255, 255)
    # ''' normalized cross correlation '''
    # template = gkern(l=10, sig=5)
    # nrmcrimg = normxcorr2(template, mask)
    # ''''''''''''''''''''''''''''''''''''
    # a = nrmcrimg
    # b = 2 * ((a - a.min()) / (a.max() - a.min())) - 1
    # b = (b - b.min()) / (b.max() - b.min())
    # mask = np.asarray(b < 0.50)
    # mask = (mask * 255).astype('uint8')

    ##### masking techinique for dots on mini
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # mask = cv2.inRange(gray, 5, 55)
    print("frame3",frame.shape)
    gray = frame[:,:,1] ### use only the green channel
    # cv2.imshow("origin",frame)
    im_blur_3 = cv2.GaussianBlur(gray,(3,3),5)
    # cv2.imshow("im_blur_3",im_blur_3)
    im_blur_8 = cv2.GaussianBlur(gray, (15,15),5)
    # cv2.imshow("im_blur_8",im_blur_8)
    im_blur_sub = im_blur_8 - im_blur_3 + 128
    # cv2.imshow("im_blur_sub",im_blur_sub)
    mask = cv2.inRange(im_blur_sub, 140, 255)
    # cv2.imshow("mask sub",mask)
    mask[:,:45] = 0
    mask[:,-45:] = 0
    # for i in range(45): #裁切1/7
    #     for j in range(240):
    #         if(mask[j][i]==255):
    #             mask[j][i]=0
    # for i in range(320-45,320):
    #     for j in range(240):
    #         if(mask[j][i]==255):
    #             mask[j][i]=0
    # cv2.imshow('test',mask)

    # ''' normalized cross correlation '''
    template = gkern(l=20, sig=3)
    nrmcrimg = normxcorr2(template, mask)
    # ''''''''''''''''''''''''''''''''''''
    a = nrmcrimg
    mask = np.asarray(a > 0.1)
    mask = (mask).astype('uint8')

    return mask


def marker_center(mask, frame):

    ''' first method '''
    # RESCALE = setting.RESCALE
    # areaThresh1=30/RESCALE**2
    # areaThresh2=1920/RESCALE**2
    # MarkerCenter = []
    # contours=cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours[1])<25:  # if too little markers, then give up
    #     print("Too less markers detected: ", len(contours))
    #     return MarkerCenter
    # for contour in contours[1]:
    #     x,y,w,h = cv2.boundingRect(contour)
    #     AreaCount=cv2.contourArea(contour)
    #     # print(AreaCount)
    #     if AreaCount>areaThresh1 and AreaCount<areaThresh2 and abs(np.max([w, h]) * 1.0 / np.min([w, h]) - 1) < 1:
    #         t=cv2.moments(contour)
    #         # print("moments", t)
    #         # MarkerCenter=np.append(MarkerCenter,[[t['m10']/t['m00'], t['m01']/t['m00'], AreaCount]],axis=0)
    #         mc = [t['m10']/t['m00'], t['m01']/t['m00']]
    #         # if t['mu11'] < -100: continue
    #         MarkerCenter.append(mc)
    #         # print(mc)
    #         cv2.circle(frame, (int(mc[0]), int(mc[1])), 10, ( 0, 0, 255 ), 2, 6);

    ''' second method '''
    img3 = mask
    cv2.imshow("mask_first",mask*255)
    neighborhood_size = 10
    # threshold = 40 # for r1.5
    threshold = 0 # for mini
    data_max = maximum_filter(img3, neighborhood_size)
    # cv2.imshow("data_max",data_max*255)
    maxima = (img3 == data_max)
    # cv2.imshow("maxima",np.uint8(maxima)*255)
    data_min = minimum_filter(img3, neighborhood_size)
    # cv2.imshow("data_min",data_min*255)
    diff = ((data_max - data_min) > threshold)
    # cv2.imshow("diff_minmax",np.uint8(diff)*255)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    MarkerCenter = np.array(ndimage.center_of_mass(img3, labeled, range(1, num_objects + 1)))
    MarkerCenter[:, [0, 1]] = MarkerCenter[:, [1, 0]]
    for i in range(MarkerCenter.shape[0]):
        x0, y0 = int(MarkerCenter[i][0]), int(MarkerCenter[i][1])
        cv2.circle(mask, (x0, y0), color=(0, 0, 255), radius=1, thickness=1)
    # cv2.imshow("mask_minmax",mask*255)
    return MarkerCenter

def draw_flow(frame, flow):
    Ox, Oy, Cx, Cy, Occupied = flow
    # print(len(Ox),len(Ox[0]))
    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx**2 + dy**2)
    print (dnet * 0.075, '\n')


    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            pt1 = (int(Ox[i][j]), int(Oy[i][j]))
            pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
            color = (0, 0, 255)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.25)

def draw_flow_contact(frame, flow, final_list):
    Ox, Oy, Cx, Cy, Occupied = flow
    # print(len(Ox),len(Ox[0]))
    dx = np.mean(np.abs(np.asarray(Ox) - np.asarray(Cx)))
    dy = np.mean(np.abs(np.asarray(Oy) - np.asarray(Cy)))
    dnet = np.sqrt(dx**2 + dy**2)
    print (dnet * 0.075, '\n')


    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            if(final_list[i][j]==1):
                pt1 = (int(Ox[i][j]), int(Oy[i][j]))
                pt2 = (int(Cx[i][j] + K * (Cx[i][j] - Ox[i][j])), int(Cy[i][j] + K * (Cy[i][j] - Oy[i][j])))
                color = (0, 0, 255)
                if Occupied[i][j] <= -1:
                    color = (127, 127, 255)
                cv2.arrowedLine(frame, pt1, pt2, color, 2,  tipLength=0.25)

def draw_flow_uv(frame, flow, u, v, final_list):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            # if(final_list[i][j]==1):
            pt = (int(Ox[i][j]), int(Oy[i][j]))
            ptu = (int(u[i+j])*6+int(Ox[i][j]), int(Oy[i][j])) 
            ptv = (int(Ox[i][j]),int(v[i+j])*6+int(Oy[i][j]))
            print(u[i+j],v[i+j])
            coloru = (255, 0, 0)
            colorv = (0, 255, 0)
            if Occupied[i][j] <= -1:
                color = (127, 127, 255)
            cv2.arrowedLine(frame, pt, ptu, coloru, 2,  tipLength=0.25)
            cv2.arrowedLine(frame, pt, ptv, colorv, 2,  tipLength=0.25)

def draw_flow_uv_contact(frame, flow, u, v, final_list):
    Ox, Oy, Cx, Cy, Occupied = flow
    K = 1
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            if(final_list[i][j]==1):
                pt = (int(Ox[i][j]), int(Oy[i][j]))
                ptu = (int(u[i+j])*6+int(Ox[i][j]), int(Oy[i][j])) 
                ptv = (int(Ox[i][j]),int(v[i+j])*6+int(Oy[i][j]))
                print(u[i+j],v[i+j])
                coloru = (255, 0, 0)
                colorv = (0, 255, 0)
                if Occupied[i][j] <= -1:
                    color = (127, 127, 255)
                cv2.arrowedLine(frame, pt, ptu, coloru, 2,  tipLength=0.25)
                cv2.arrowedLine(frame, pt, ptv, colorv, 2,  tipLength=0.25)


def warp_perspective(img):

    TOPLEFT = (175,230)
    TOPRIGHT = (380,225)
    BOTTOMLEFT = (10,410)
    BOTTOMRIGHT = (530,400)

    WARP_W = 215
    WARP_H = 215

    points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
    points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])

    matrix=cv2.getPerspectiveTransform(points1,points2)

    result = cv2.warpPerspective(img, matrix, (WARP_W,WARP_H))

    return result


def init_HSR(img):
    DIM=(640, 480)
    img = cv2.resize(img, DIM)

    K=np.array([[225.57469247811056, 0.0, 280.0069549918857], [0.0, 221.40607131318117, 294.82435570493794], [0.0, 0.0, 1.0]])
    D=np.array([[0.7302503082668154], [-0.18910060205317372], [-0.23997727800712282], [0.13938490908400802]])
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warp_perspective(undistorted_img)
