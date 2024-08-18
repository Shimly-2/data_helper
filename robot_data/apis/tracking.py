import copy

import find_marker
import numpy as np
import cv2
import time
import marker_detection
import scipy
import math
from scipy.interpolate import griddata
import sys
sys.path.append('/home/jackeyjin/gsrobotics')
from gelsight import gs3drecon
import setting
#import matplotlib.pyplot as plt
import pickle as pk
#from utils.live_ximea import GelSight
from camera_calibration import warp_perspective
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import enum

mtxl = [[2.25213449e+03, 0.00000000e+00, 1.65017047e+02]
 ,[0.00000000e+00, 2.11441402e+03, 1.16358187e+02]
 ,[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
distl = [[ 4.31270333e+01, -6.62123007e+03, 2.94678406e-02, 1.88904778e-02, -2.57466149e+04]]

#
# 2D integration via Poisson solver
#
def poisson_reconstruct(grady, gradx, boundarysrc):
    # Laplacian
    gyy = grady[1:,:-1] - grady[:-1,:-1]
    gxx = gradx[:-1,1:] - gradx[:-1,:-1]
    f = np.zeros(boundarysrc.shape)
    f[:-1,1:] += gxx
    f[1:,:-1] += gyy
    # Boundary image
    boundary = boundarysrc.copy()
    boundary[1:-1,1:-1] = 0

    # Subtract boundary contribution
    f_bp = -4*boundary[1:-1,1:-1] + boundary[1:-1,2:] + boundary[1:-1,0:-2] + boundary[2:,1:-1] + boundary[0:-2,1:-1]
    f = f[1:-1,1:-1] - f_bp

    # Discrete Sine Transform
    #tt = np.zeros(boundarysrc.shape)
    #fsin = np.zeros(boundarysrc.shape)
    #cv2.dft(f, tt)
    #cv2.dft(tt.T, fsin)
    #cv2.namedWindow('dft')
    #cv2.imshow('dft',fsin)
    #cv2.waitKey()
    tt = scipy.fftpack.dst(f, norm='ortho')
    fsin = scipy.fftpack.dst(tt.T, norm='ortho').T
    # Eigenvalues
    (x,y) = np.meshgrid(range(1,f.shape[1]+1), range(1,f.shape[0]+1), copy=True)
    denom = (2*np.cos(math.pi*x/(f.shape[1]+2))-2) + (2*np.cos(math.pi*y/(f.shape[0]+2)) - 2)
    f = fsin/denom
    # Inverse Discrete Sine Transform
    #img_tt = np.zeros(f.shape)
    #cv2.idft(f, tt)
    #cv2.idft(tt.T, img_tt)
    tt = scipy.fftpack.idst(f, norm='ortho')
    img_tt = scipy.fftpack.idst(tt.T, norm='ortho').T
    print('tt shape = ', tt.shape)
    # New center + old boundary
    result = boundary
    result[1:-1,1:-1] = img_tt
    return result

class RGB2NormNetR1(nn.Module):
    def __init__(self):
        super(RGB2NormNetR1, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.fc7 = nn.Linear(8, 2)
        self.drop_layer = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc2(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc3(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc4(x))
        x = self.drop_layer(x)
        x = F.tanh(self.fc5(x))
        x = F.tanh(self.fc6(x))
        x = self.fc7(x)
        return x


''' nn architecture for r1.5 and mini '''
class RGB2NormNetR15(nn.Module):
    def __init__(self):
        super(RGB2NormNetR15, self).__init__()
        input_size = 5
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,2)
        self.drop_layer = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = self.fc4(x)
        return x

# creating enumerations using class
class Finger(enum.Enum):
    R1 = 1
    R15 = 2
    DIGIT = 3
    MINI = 4

def dilate(img, ksize=5, iter=1):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(img, kernel, iterations=iter)

def erode(img, ksize=5):
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.erode(img, kernel, iterations=1)
def interpolate_grad(img, mask):
    # mask = (soft_mask > 0.5).astype(np.uint8) * 255
    # pixel around markers
    mask_around = (dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
    # mask_around = mask == 0
    mask_around = mask_around.astype(np.uint8)
    # cv2.imshow("mask_around", mask_around*1.)

    x, y = np.arange(img.shape[0]), np.arange(img.shape[1])
    yy, xx = np.meshgrid(y, x)

    # mask_zero = mask == 0
    mask_zero = mask_around == 1
    # cv2.imshow("mask_zero", mask_zero*1.)

    # if np.where(mask_zero)[0].shape[0] != 0:
    #     print ('interpolating')
    mask_x = xx[mask_around == 1]
    mask_y = yy[mask_around == 1]
    points = np.vstack([mask_x, mask_y]).T
    values = img[mask_x, mask_y]
    markers_points = np.vstack([xx[mask != 0], yy[mask != 0]]).T
    method = "nearest"
    # method = "linear"
    # method = "cubic"
    x_interp = griddata(points, values, markers_points, method=method)
    x_interp[x_interp != x_interp] = 0.0
    ret = img.copy()
    ret[mask != 0] = x_interp
    # else:
    #     ret = img
    return ret

def find_markern(gray):
    mask = cv2.inRange(gray, 0, 70)
    # kernel = np.ones((2, 2), np.uint8)
    # dilation = cv2.dilate(mask, kernel, iterations=1)
    return mask

def demark(gx, gy, markermask):
    # mask = find_marker(img)
    gx_interp = interpolate_grad(gx.copy(), markermask)
    gy_interp = interpolate_grad(gy.copy(), markermask)
    return gx_interp, gy_interp

class Reconstruction3D:
    def __init__(self, finger):
        self.finger = finger
        self.cpuorgpu = "cpu"
        self.dm_zero_counter = 0
        self.dm_zero = np.zeros((240, 320))
        pass

    def load_nn(self, net_path, cpuorgpu):

        self.cpuorgpu = cpuorgpu
        device = torch.device(cpuorgpu)

        if not os.path.isfile(net_path):
            print('Error opening ', net_path, ' does not exist')
            return

        print('self.finger = ', self.finger)
        if self.finger == Finger.R1:
            print('calling nn R1...')
            net = RGB2NormNetR1().float().to(device)
        elif self.finger == Finger.R15:
            print('calling nn R15...')
            net = RGB2NormNetR15().float().to(device)
        else:
            net = RGB2NormNetR15().float().to(device)

        if cpuorgpu=="cuda":
            ### load weights on gpu
            # net.load_state_dict(torch.load(net_path))
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage.cuda(0))
            net.load_state_dict(checkpoint['state_dict'])
        else:
            ### load weights on cpu which were actually trained on gpu
            checkpoint = torch.load(net_path, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])

        self.net = net

        return self.net

    def get_depthmap(self, frame, mask_markers, cm=None):
        MARKER_INTERPOLATE_FLAG = mask_markers

        ''' find contact region '''
        # cm, cmindx = find_contact_mask(f1, f0)
        ###################################################################
        ### check these sizes
        ##################################################################
        if (cm is None):
            cm, cmindx = np.ones(frame.shape[:2]), np.where(np.ones(frame.shape[:2]))
        imgh = frame.shape[:2][0]
        imgw = frame.shape[:2][1]

        if MARKER_INTERPOLATE_FLAG:
            ''' find marker mask '''
            markermask = find_markern(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            cm = ~markermask
            '''intersection of cm and markermask '''
            # cmmm = np.zeros(img.shape[:2])
            # ind1 = np.vstack(np.where(cm)).T
            # ind2 = np.vstack(np.where(markermask)).T
            # ind2not = np.vstack(np.where(~markermask)).T
            # ind3 = matching_rows(ind1, ind2)
            # cmmm[(ind3[:, 0], ind3[:, 1])] = 1.
            cmandmm = (np.logical_and(cm, markermask)).astype('uint8')
            cmandnotmm = (np.logical_and(cm, ~markermask)).astype('uint8')

        ''' Get depth image with NN '''
        nx = np.zeros(frame.shape[:2])
        ny = np.zeros(frame.shape[:2])
        dm = np.zeros(frame.shape[:2])

        ''' ENTIRE CONTACT MASK THRU NN '''
        # if np.where(cm)[0].shape[0] != 0:
        rgb = frame[np.where(cm)] / 255
        # rgb = diffimg[np.where(cm)]
        pxpos = np.vstack(np.where(cm)).T
        # pxpos[:, [1, 0]] = pxpos[:, [0, 1]] # swapping
        pxpos[:, 0], pxpos[:, 1] = pxpos[:, 0] / imgh, pxpos[:, 1] / imgw
        # the neural net was trained using height=320, width=240
        # pxpos[:, 0] = pxpos[:, 0] / ((320 / imgh) * imgh)
        # pxpos[:, 1] = pxpos[:, 1] / ((240 / imgw) * imgw)

        features = np.column_stack((rgb, pxpos))
        features = torch.from_numpy(features).float().to(self.cpuorgpu)
        with torch.no_grad():
            self.net.eval()
            out = self.net(features)

        nx[np.where(cm)] = out[:, 0].cpu().detach().numpy()
        ny[np.where(cm)] = out[:, 1].cpu().detach().numpy()
        # print(nx.min(), nx.max(), ny.min(), ny.max())
        # nx = 2 * ((nx - nx.min()) / (nx.max() - nx.min())) -1
        # ny = 2 * ((ny - ny.min()) / (ny.max() - ny.min())) -1
        # print(nx.min(), nx.max(), ny.min(), ny.max())

        '''OPTION#1 normalize gradient between [a,b]'''
        # a = -5
        # b = 5
        # gx = (b-a) * ((gx - gx.min()) / (gx.max() - gx.min())) + a
        # gy = (b-a) * ((gy - gy.min()) / (gy.max() - gy.min())) + a
        '''OPTION#2 calculate gx, gy from nx, ny. '''
        # nz = np.sqrt(1 - nx ** 2 - ny ** 2)
        # if np.isnan(nz).any():
        #     print ('nan found')
        # nz[np.where(np.isnan(nz))] = 0
        # print('nz',nz)
        # gx = nx / nz
        # gy = ny / nz
        gx = nx / 0.73
        gy = ny / 0.73

        if MARKER_INTERPOLATE_FLAG:
            # gx, gy = interpolate_gradients(gx, gy, img, cm, cmmm)
            dilated_mm = dilate(markermask, ksize=3, iter=2)
            gx_interp, gy_interp = demark(gx, gy, dilated_mm)
        else:
            gx_interp, gy_interp = gx, gy

        # print (gx.min(), gx.max(), gy.min(), gy.max())
        # nz = np.sqrt(1 - nx ** 2 - ny ** 2) ### normalize normals to get gradients for poisson
        #print(gy_interp.shape)
        boundary = np.zeros((imgh, imgw))

        dm = poisson_reconstruct(gx_interp, gy_interp, boundary)
        dm = np.reshape(dm, (imgh, imgw))
        #print(dm.shape)
        # cv2.imshow('dm',dm)

        ''' remove initial zero depth '''
        if self.dm_zero_counter < 50:
            self.dm_zero += dm
            print ('zeroing depth. do not touch the gel!')
            if self.dm_zero_counter == 49:
                self.dm_zero /= self.dm_zero_counter
        else:
            print ('touch me!')
        self.dm_zero_counter += 1
        dm = dm - self.dm_zero
        # print(dm.min(), dm.max())

        ''' ENTIRE MASK. GPU OPTIMIZED VARIABLES. '''
        # if np.where(cm)[0].shape[0] != 0:
        ### Run things through NN. FAST!!??
        # pxpos = np.vstack(np.where(cm)).T
        # features = np.zeros((len(pxpos), 5))
        # get_features(img, pxpos, features, imgw, imgh)
        # features = torch.from_numpy(features).float().to(device)
        # with torch.no_grad():
        #     net.eval()
        #     out = net(features)
        # # Create gradient images and do reconstuction
        # gradx = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady = torch.from_numpy(np.zeros_like(cm, dtype=np.float32)).to(device)
        # grady[pxpos[:, 0], pxpos[:, 1]] = out[:, 0]
        # gradx[pxpos[:, 0], pxpos[:, 1]] = out[:, 1]
        # # dm = poisson_reconstruct_gpu(grady, gradx, denom).cpu().numpy()
        # dm = cv2.resize(poisson_reconstruct(grady, gradx, denom).cpu().numpy(), (640, 480))
        # dm = cv2.resize(dm, (imgw, imgh))
        # # dm = np.clip(dm / img.max(), 0, 1)
        # # dm = 255 * dm
        # # dm = dm.astype(np.uint8)

        ''' normalize gradients for plotting purpose '''
        #print(gx.min(), gx.max(), gy.min(), gy.max())
        gx = (gx - gx.min()) / (gx.max() - gx.min())
        gy = (gy - gy.min()) / (gy.max() - gy.min())
        gx_interp = (gx_interp - gx_interp.min()) / (gx_interp.max() - gx_interp.min())
        gy_interp = (gy_interp - gy_interp.min()) / (gy_interp.max() - gy_interp.min())

        return dm


def find_cameras():
    # checks the first 10 indexes.
    index = 0
    arr = []
    if os.name == 'nt':
        cameras = find_cameras_windows()
        return cameras
    i = 10
    while i >= 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            command = 'v4l2-ctl -d ' + str(index) + ' --info'
            is_arducam = os.popen(command).read()
            if is_arducam.find('Arducam') != -1 or is_arducam.find('Mini') != -1:
                arr.append(index)
            cap.release()
        index += 1
        i -= 1

    return arr

def find_cameras_windows():
    # checks the first 10 indexes.
    index = 0
    arr = []
    idVendor = 0xC45
    idProduct = 0x636D
    import usb.core
    import usb.backend.libusb1
    backend = usb.backend.libusb1.get_backend(
        find_library=lambda x: "libusb_win/libusb-1.0.dll"
    )
    dev = usb.core.find(backend=backend, find_all=True)
    # loop through devices, printing vendor and product ids in decimal and hex
    for cfg in dev:
        #print('Decimal VendorID=' + hex(cfg.idVendor) + ' & ProductID=' + hex(cfg.idProduct) + '\n')
        if cfg.idVendor == idVendor and cfg.idProduct == idProduct:
            arr.append(index)
        index += 1

    return arr


def resize_crop_mini(img, imgw, imgh):
    # resize, crop and resize back
    img = cv2.resize(img, (895, 672))  # size suggested by janos to maintain aspect ratio
    # border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))  # remove 1/7th of border from each size
    # img = img[border_size_x:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    img = img[:, :-1]  # remove last column to get a popular image resolution
    img = cv2.resize(img, (imgw, imgh))  # final resize for 3d
    return img

def trim(img):
    img[img<0] = 0
    img[img>255] = 255



def compute_tracker_gel_stats(thresh):
    numcircles = 9 * 7
    mmpp = .0625
    true_radius_mm = .5
    true_radius_pixels = true_radius_mm / mmpp
    circles = np.where(thresh)[0].shape[0]
    circlearea = circles / numcircles
    radius = np.sqrt(circlearea / np.pi)
    radius_in_mm = radius * mmpp
    percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2)
    return radius_in_mm, percent_coverage*100.

def calibrate_img(img):
    fs = cv2.FileStorage('demos/mini_marker_tracking/calibration_parameter_L.yaml', cv2.FileStorage_READ)
    mtxl = fs.getNode('camera_Matrix_L').mat()
    distl = fs.getNode('dist_Coeffs_L').mat()

    img_distort = cv2.undistort(img, np.array(mtxl), np.array(distl))
    img_diff = cv2.absdiff(img, img_distort)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtxl, distl, (w,h), 0, (w,h))
    dst = cv2.undistort(img, mtxl, distl, None, newcameramtx)

    # 剪裁图像
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def flow_calculate_global(img, flow):
    u_sum = np.zeros(63)
    v_sum = np.zeros(63)
    u_addon = list(u_sum)
    v_addon = list(v_sum)
    x_iniref, y_iniref = [], []
    x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired  = [], [], [], [], [], [], [], []
    x1_return, y1_return, x2_return, y2_return, u_return, v_return = [],[],[],[],[],[]

    Ox, Oy, Cx, Cy, Occupied = flow
    for i in range(len(Ox)):
        for j in range(len(Ox[i])):
            x2.append(Cx[i][j])
            y2.append(Cy[i][j])
            x_iniref.append(Ox[i][j])
            y_iniref.append(Oy[i][j])
    # for i in range(len(keypoints2)): 
    #     x2.append(keypoints2[i].pt[0]/self.scale)
    #     y2.append(keypoints2[i].pt[1]/self.scale)

    x2 = np.array(x2) 
    y2 = np.array(y2)
    x_initial = list(x_iniref)
    y_initial = list(y_iniref)
    u_ref = list(u_addon)
    v_ref = list(v_addon)
    

    for i in range(x2.shape[0]):
        distance = list(((np.array(x_initial) - x2[i])**2 + (np.array(y_initial) - y2[i])**2))
        min_index = distance.index(min(distance))  
        u_temp = x2[i] - x_initial[min_index] 
        v_temp = y2[i] - y_initial[min_index] 
        shift_length = np.sqrt(u_temp**2+v_temp**2)
        # print 'length',shift_length

        # print xy2.shape,min_index,len(distance)
        if shift_length < 7:
            x1_paired.append(x_initial[min_index]-u_ref[min_index])
            y1_paired.append(y_initial[min_index]-v_ref[min_index])
            x2_paired.append(x2[i])
            y2_paired.append(y2[i])
            u.append(u_temp + u_ref[min_index])
            v.append(v_temp + v_ref[min_index])
            # sign = self.ROI[y2[i].astype(np.uint16),x2[i].astype(np.uint16)]
            # x1_return.append((x_initial[min_index]-u_ref[min_index])*sign)
            # y1_return.append((y_initial[min_index]-v_ref[min_index])*sign)
            # x2_return.append((x2[i])*sign)
            # y2_return.append((y2[i])*sign)
            # u_return.append((u_temp + u_ref[min_index])*sign)
            # v_return.append((v_temp + v_ref[min_index])*sign)
            del x_initial[min_index], y_initial[min_index], u_ref[min_index], v_ref[min_index]   

        
    # print('len:',len(x_iniref), len(x2_paired))
    x_iniref = list(x2_paired) 
    y_iniref = list(y2_paired)
    u_addon = list(u)
    v_addon = list(v)
    refresh = False 
    # print(np.array(y2_paired).astype(np.uint16),np.array(x2_paired).astype(np.uint16),np.array(range(len(x2_paired))))
    # inbound_check = img[np.array(y2_paired).astype(np.uint16),np.array(x2_paired).astype(np.uint16)]*np.array(range(len(x2_paired)))

    # final_list = list(set(inbound_check)- set([0]))
    # final_list = list(set(x2_paired)- set([0]))
    # x1_return = np.array(x1_paired)[final_list]
    # y1_return = np.array(y1_paired)[final_list]
    # x2_return = np.array(x2_paired)[final_list]
    # y2_return = np.array(y2_paired)[final_list]
    # u_return = np.array(u)[final_list]
    # v_return = np.array(v)[final_list]

    x1_return = np.array(x1_paired)
    y1_return = np.array(y1_paired)
    x2_return = np.array(x2_paired)
    y2_return = np.array(y2_paired)
    u_return = np.array(u)
    v_return = np.array(v)

    return x1_return, y1_return, x2_return, y2_return, u_return, v_return

def estimate_uv(tran_matrix,x1,y1,u_sum,v_sum,x2,y2):
    theta = np.arcsin(tran_matrix[1,0])
    x1_select = np.array(x1)
    y1_select = np.array(y1)
    u_select = u_sum
    v_select = v_sum

    u_mean = np.mean(u_select)
    v_mean = np.mean(v_select)
    x_mean = np.mean(x1_select)
    y_mean = np.mean(y1_select)

    u_estimate = u_mean + theta*(y_mean - np.array(y2))
    v_estimate = v_mean + theta*(np.array(x2)-x_mean)

    return u_estimate, v_estimate

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

def make_thre_mask(im_cal):
    thre_image = np.zeros(im_cal.shape,dtype = np.uint8)
    previous_mask = np.zeros(im_cal.shape,dtype = np.uint8)
    for i in range(4,6,1):
        th = i/100.0
        _, mask = cv2.threshold(im_cal, th, 255, cv2.THRESH_BINARY)
        mask_expand = cv2.dilate(mask, kernal10, iterations=1)
        mask_erode = cv2.erode(mask_expand, kernal10, iterations=1)
        thre_image_origin = np.uint8(mask_erode)
        # mask_erode=np.array(mask_erode,dtype=np.uint8)
        thre_image = thre_image + (mask_erode - previous_mask)/255*i 
        # cv2.imshow('mask_erode{}'.format(i), mask_erode)
        previous_mask = mask_erode
        # cv2.imshow('threshold{}'.format(th), thre_image)
        # cv2.waitKey(0)
    # thre_image_origin = thre_image
    thre_image = thre_image + (np.ones(im_cal.shape, dtype = np.uint8) - previous_mask/255) * 65 + 10
    # cv2.imshow('thre_image', thre_image)
    # cv2.waitKey(0)
    return thre_image, thre_image_origin

def creat_mask(im_cal,thre_image):
    thresh1 = (im_cal < thre_image).astype(np.uint8)

    temp1 = cv2.dilate(thresh1, kernal9, iterations=1)
    temp2 = cv2.erode(temp1, kernal8, iterations=1)
    # cv2.imshow('thresh1',temp2)
    # cv2.waitKey(1)
    final_image1 = cv2.dilate(temp2, kernal5, iterations=1)
    # final_image2 = cv2.dilate(final_image1, self.kernal9, iterations=1)
    # cv2.imshow('creat_mask',final_image1*255)
    # cv2.waitKey(1)
    return (1-final_image1)*255

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

def contact_detection_new(frame, origin):
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.5, 0.5, 0.])


def main(argv):

    imgw = 320
    imgh = 240

    USE_VIDEO = True
    USE_LIVE_R1 = False
    calibrate = False
    border_size = 25

    outdir = './TEST/'
    SAVE_VIDEO_FLAG = True
    SAVE_ONE_IMG_FLAG = False
    SAVE_DATA_FLAG = False
    MASK_MARKERS_FLAG = True

    net_file_path = 'nnmini.pt'
    path = 'examples'
    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)
    gpuorcpu = "cpu"
    
    nn = Reconstruction3D(gs3drecon.Finger.R15)
    net = nn.load_nn(net_path, gpuorcpu)

    x1, y1, x2, y2, u, v = [],[],[],[],[],[]
    slip_monitor = {}
    previous_u_sum = np.array([0])
    previous_v_sum = np.array([0])

    if SAVE_ONE_IMG_FLAG:
        sn = input('Please enter the serial number of the gel \n')
        #sn = str(5)
        viddir = outdir + 'vids/'
        imgdir = outdir + 'imgs/'
        resultsfile = outdir + 'marker_qc_results.txt'
        vidfile = viddir + sn + '.avi'
        imgonlyfile = imgdir + sn + '.png'
        maskfile = imgdir + 'mask_' + sn + '.png'
        # check to see if the directory exists, if not create it
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        if not os.path.exists(viddir):
            os.mkdir(viddir)
        if not os.path.exists(imgdir):
            os.mkdir(imgdir)

    if SAVE_DATA_FLAG:
        datadir = outdir + 'data'
        datafilename = datadir + 'marker_locations.txt'
        datafile = open(datafilename,"a")

    # if len(sys.argv) > 1:
    #     if sys.argv[1] == 'calibrate':
    #         calibrate = True


    if USE_LIVE_R1:
        gs = GelSight(0)
        WHILE_COND = 1
    else:
        # cameras = find_cameras()
        # cap = cv2.VideoCapture(cameras[0])
        # cap = cv2.VideoCapture('http://pi:robits@raspiatgelsightinc.local:8080/?action=stream')
        cap = cv2.VideoCapture('/home/jackeyjin/gsrobotics/demos/mini_marker_tracking/3dnnlive3.mp4')
        WHILE_COND = cap.isOpened()

    # set the format into MJPG in the FourCC format
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))

    # Resize scale for faster image processing
    setting.init()
    RESCALE = setting.RESCALE

    if SAVE_VIDEO_FLAG:
        # Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fourcc = cv2.VideoWriter_fourcc('p', 'n', 'g', ' ')
        # fourcc = cv2.VideoWriter_fourcc(*'jp2','')
        out = cv2.VideoWriter("force_test2_single_flow.mp4", fourcc, 30, (imgw, imgh), isColor=True)

    frame0 = None
    ret, origin = cap.read()
    origin = calibrate_img(origin)
    origin = resize_crop_mini(origin, imgw, imgh)

    counter = 0
    while 1:
        if counter<50:
            ret, frame = cap.read()
            print ('flush black imgs')

            if counter == 48:
                ret, frame = cap.read()

                frame = calibrate_img(frame)
                print("frame1",frame.shape)
                frame = resize_crop_mini(frame, imgw, imgh)
                print("frame2",frame.shape)
                ### find marker masks
                mask = marker_detection.find_marker(frame)
                ### find marker centers
                mc = marker_detection.marker_center(mask, frame)
                break

            counter += 1

    counter = 0


    mccopy = mc
    mc_sorted1 = mc[mc[:,0].argsort()]
    mc1 = mc_sorted1[:setting.N_]
    mc1 = mc1[mc1[:,1].argsort()]

    mc_sorted2 = mc[mc[:,1].argsort()]
    mc2 = mc_sorted2[:setting.M_]
    mc2 = mc2[mc2[:,0].argsort()]


    """
    N_, M_: the row and column of the marker array
    x0_, y0_: the coordinate of upper-left marker
    dx_, dy_: the horizontal and vertical interval between adjacent markers
    """
    N_= setting.N_
    M_= setting.M_
    fps_ = setting.fps_
    x0_ = np.round(mc1[0][0])
    y0_ = np.round(mc1[0][1])
    dx_ = mc2[1, 0] - mc2[0, 0]
    dy_ = mc1[1, 1] - mc1[0, 1]

    print ('x0:',x0_,'\n', 'y0:', y0_,'\n', 'dx:',dx_,'\n', 'dy:', dy_)

    radius ,coverage = compute_tracker_gel_stats(mask)

    if SAVE_ONE_IMG_FLAG:
        fresults = open(resultsfile, "a")
        fresults.write(f"{sn} {float(f'{dx_:.2f}')} {float(f'{dy_:.2f}')} {float(f'{radius*2:.2f}')} {float(f'{coverage:.2f}')}\n")


    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('frame', 320*3, 240*3)
    # cv2.resizeWindow('mask', 320*3, 240*3)
    # Create Mathing Class

    m = find_marker.Matching(N_,M_,fps_,x0_,y0_,dx_,dy_)

    frameno = 0
    # final_list_num = 0
    # final_listn = [[0]*9 for _ in range(7)]
    try:
        while (WHILE_COND):

            if USE_LIVE_R1:
                gs.cam.get_image(gs.img)
                frame = gs.img.get_image_data_numpy()
            else:
                ret, frame = cap.read()
                if not(ret):
                    break
            final_listn = [[0]*9 for _ in range(7)]
            final_list_num = 0
            frame = calibrate_img(frame)
            ##########################
            # resize (or unwarp)
            # frame = cv2.resize(frame, (imgw,imgh))
            # frame = frame[10:-10,5:-10] # for R1.5
            # frame = frame[border_size:imgh - border_size, border_size:imgw - border_size] # for mini
            frame = resize_crop_mini(frame, imgw, imgh)
            raw_img = copy.deepcopy(frame)
            raw_img0 = frame.copy()

            dm = nn.get_depthmap(raw_img, MASK_MARKERS_FLAG)

            # frame = frame[55:,:]
            # frame = cv2.resize(frame, (imgw, imgh))


            ''' EXTRINSIC calibration ... 
            ... the order of points [x_i,y_i] | i=[1,2,3,4], are same 
            as they appear in plt.imshow() image window. Put them in 
            clockwise order starting from the topleft corner'''
            # frame = frame[30:400, 70:400]
            # frame = warp_perspective(frame, [[35, 15], [320, 15], [290, 360], [65, 360]], output_sz=frame.shape[:2])   # params for small dots
            # frame = warp_perspective(frame, [[180, 130], [880, 130], [800, 900], [260, 900]], output_sz=(640,480)) # org. img size (1080x1080)

            ### find marker masks
            mask = marker_detection.find_marker(frame)
            ### find marker centers
            mc = marker_detection.marker_center(mask, frame)

            if calibrate == False:
                tm = time.time()
                ### matching init
                m.init(mc)

                ### matching
                m.run()
                # print(time.time() - tm)

                ### matching result
                """
                output: (Ox, Oy, Cx, Cy, Occupied) = flow
                    Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
                    Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
                    Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                        e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
                """
                flow = m.get_flow()

                if frame0 is None:
                    frame0 = frame.copy()
                    frame0 = cv2.GaussianBlur(frame0, (int(63), int(63)), 0)

                # diff = (frame * 1.0 - frame0) * 4 + 127
                # trim(diff)
                # cv2.imshow('diffnew',diff)

                # # draw flow
                # marker_detection.draw_flow(frame, flow)

                # calculate flow
                x1, y1, x2, y2, u, v = flow_calculate_global(frame,flow)
                # print(u, v)
                u_sum = np.array(u)
                v_sum = np.array(v)
                x2_center = np.expand_dims(np.array(x2),axis = 1)
                y2_center = np.expand_dims(np.array(y2),axis = 1)
                x1_center = np.expand_dims(np.array(x1),axis = 1)
                y1_center = np.expand_dims(np.array(y1),axis = 1)
                p2_center = np.expand_dims(np.concatenate((x2_center,y2_center),axis = 1),axis = 0)
                p1_center = np.expand_dims(np.concatenate((x1_center,y1_center),axis = 1),axis = 0)
                # print len(final_list)
                # if len(final_list)>1:
                tran_matrix, _ = cv2.estimateAffinePartial2D(p1_center,p2_center,False)
                # print('tran_matrix: ',tran_matrix)

                if tran_matrix is not None:
                    u_estimate, v_estimate = estimate_uv(tran_matrix,x1,y1,u_sum,v_sum,x2,y2)
                    # print('u_estimate:',u_estimate.shape,'v_estimate:',v_estimate)
                    vel_diff = np.sqrt((u_estimate - u_sum)**2 + (v_estimate - v_sum)**2)
                    u_diff = u_estimate - u_sum
                    v_diff = v_estimate - v_sum
                if np.abs(np.mean(v_sum)) > np.abs(np.mean(u_sum)) + 2:
                    thre_slip_dis = 3.5
                else:
                    thre_slip_dis = 4.5

                    numofslip = np.sum(vel_diff > thre_slip_dis)
                    # print ("number of marker: ", numofslip, 'vel_diff: ', vel_diff)
    
                    abs_change_u = np.abs(previous_u_sum - np.mean(u_sum))
                    abs_change_v = np.abs(previous_v_sum - np.mean(v_sum))
                    abs_change = np.sqrt(abs_change_u**2+abs_change_v**2)
                    # diff_img_sum = np.sum(np.abs(previous_image.astype(np.int16) - final_image.astype(np.int16)))
                    # print ('abs_change_u: ', abs_change_u, 'abs_change_v: ', abs_change_v)
                    
                    # if (abs_change_u < 0.05 + int(self.static_flag)*0.05 and abs_change_v < 0.05 + int(self.static_flag)*0.05):
                    #     self.slip_indicator = False
                    #     self.static_flag = True
                    # else:
                    thre_slip_num = 7
                    slip_indicator = numofslip > thre_slip_num
                    static_flag = False
                    
                    # raw_input("Press Enter to continue...")
                    # if self.showimage:
                    #     self.dispOpticalFlow(im_cal_show,x2,y2)
                if tran_matrix is None:
                    slip_monitor['values'] = [np.mean(np.array(u)),np.mean(np.array(v)),np.arcsin(self.tran_matrix[1,0])/np.pi*180]
                    # if len(final_list) >3:
                    #     self.slip_indicator = True
                else:
                    slip_monitor['values'] = [np.mean(np.array(u)),np.mean(np.array(v)),0.]

                slip_monitor['name'] = str(slip_indicator)
                # print('slip_monitor: ', slip_monitor)

                previous_slip = slip_indicator
                previous_u_sum = np.mean(np.array(u))
                previous_v_sum = np.mean(np.array(v))

                
                if slip_indicator: 
                    print("slip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
                    slip_indicator = False
                    break

                # method 1
                # frame2 = calibrate_img(raw_img)
                b,g,r=cv2.split(raw_img0)
                # frame1 = rgb2gray(frame)
                # frame2 = contact_detection(b)
                frame2 = contact_detection_new(raw_img0, origin)
                frame2 = cv2.medianBlur(frame2,25)
                contours = cv2.findContours(frame2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                fill_image = np.zeros((240, 320),dtype=np.uint8) 

                if contours is not None:
                    area = []
                    topk_contours =[]
                    x, y = frame2.shape
                    for i in range(len(contours)):
                        # 对每一个连通域使用一个掩码模板计算非0像素(即连通域像素个数)
                        single_masks = np.zeros((x, y),dtype=np.uint8) 
                        fill_image = cv2.fillConvexPoly(single_masks, np.array(contours[i],dtype = np.int), 255)
                        pixels = cv2.countNonZero(fill_image)
                        if(pixels<1500):
                            fill_image = cv2.fillConvexPoly(single_masks, contours[i], 0)
                        else:
                            area.append(pixels)
                        frame2 = cv2.bitwise_or(single_masks,fill_image)
                    # 画接触区域轮廓线####################################################
                    # if len(area): 
                    #     cv2.drawContours(frame, contours, -1, (0,255,0), 2)
                    # bitand = cv2.medianBlur(frame2,21)
                    # if(len(frame2[frame2==255])<1000):
                    #     frame[:,:,1] = frame[:,:,1]
                    # else:
                    #     frame[:,:,1] = frame[:,:,1] + bitand/7
                    ####################################################################

                    Ox, Oy, Cx, Cy, Occupied = flow
                    for i in range(len(Ox)):
                        for j in range(len(Ox[i])):
                            if(fill_image[int(Cy[i][j])][int(Cx[i][j])]==255):
                                final_listn[i][j] = 1
                    # print('final_list: ',final_listn)
                    frameuv = copy.deepcopy(frame)
                    marker_detection.draw_flow(frame, flow)
                    # marker_detection.draw_flow_contact(frame, flow, final_listn)
                    marker_detection.draw_flow_uv_contact(frameuv, flow, u_estimate, v_estimate, final_listn)


                frameno = frameno + 1

                if SAVE_DATA_FLAG:
                    Ox, Oy, Cx, Cy, Occupied = flow
                    for i in range(len(Ox)):
                        for j in range(len(Ox[i])):
                            datafile.write(
                               f"{frameno}, {i}, {j}, {Ox[i][j]:.2f}, {Oy[i][j]:.2f}, {Cx[i][j]:.2f}, {Cy[i][j]:.2f}\n")

            mask_img = np.asarray(mask)
            contours,hierarchy=cv2.findContours((mask_img*255).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #绘制标记点轮廓线 显示圆圈
            # cv2.drawContours(frame,contours,-1,(0,0,255),1,lineType=cv2.LINE_AA)

            bigframe = cv2.resize(frame, (int(frame.shape[1]*1.5), int(frame.shape[0]*1.5)))
            cv2.imshow('frame', bigframe)
            bigframeuv = cv2.resize(frameuv, (int(frameuv.shape[1]*1.5), int(frameuv.shape[0]*1.5)))
            cv2.imshow('frameuv', bigframeuv)
            bigmask = cv2.resize(mask_img*255, (int(mask_img.shape[1]*1.5), int(mask_img.shape[0]*1.5)))
            cv2.imshow('mask', bigmask)

            if SAVE_ONE_IMG_FLAG:
                cv2.imwrite(imgonlyfile, raw_img)
                cv2.imwrite(maskfile, mask*255)
                SAVE_ONE_IMG_FLAG = False

            if calibrate:
                ### Display the mask
                cv2.imshow('mask',mask_img*255)
            if SAVE_VIDEO_FLAG:
                out.write(frame)
            # print(frame.shape)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print('Interrupted!')

    ### release the capture and other stuff
    if USE_LIVE_R1:
        gs.end_process()
    else:
        cap.release()
        cv2.destroyAllWindows()
    if SAVE_VIDEO_FLAG:
        out.release()

if __name__ == "__main__":
    main(sys.argv[1:])
