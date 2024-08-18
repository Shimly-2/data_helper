import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from robot_data.apis.finger import Finger, RGB2NormNetR1, RGB2NormNetR15
import cv2
from robot_data.apis.poisson2d import poisson_reconstruct
from scipy.interpolate import griddata

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
    
    def find_markern(self, gray):
        mask = cv2.inRange(gray, 0, 70)
        # kernel = np.ones((2, 2), np.uint8)
        # dilation = cv2.dilate(mask, kernel, iterations=1)
        return mask
    
    def dilate(self, img, ksize=5, iter=1):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.dilate(img, kernel, iterations=iter)

    def erode(self, img, ksize=5):
        kernel = np.ones((ksize, ksize), np.uint8)
        return cv2.erode(img, kernel, iterations=1)
    
    def demark(self, gx, gy, markermask):
        # mask = find_marker(img)
        gx_interp = self.interpolate_grad(gx.copy(), markermask)
        gy_interp = self.interpolate_grad(gy.copy(), markermask)
        return gx_interp, gy_interp
    
    def interpolate_grad(self, img, mask):
        # mask = (soft_mask > 0.5).astype(np.uint8) * 255
        # pixel around markers
        mask_around = (self.dilate(mask, ksize=3, iter=2) > 0) & ~(mask != 0)
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

    def get_depthmap(self, net, frame, mask_markers, cm=None):
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
            markermask = self.find_markern(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
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
            net.eval()
            out = net(features)

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
            dilated_mm = self.dilate(markermask, ksize=3, iter=2)
            gx_interp, gy_interp = self.demark(gx, gy, dilated_mm)
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
        # if self.dm_zero_counter < 50:
        #     self.dm_zero += dm
        #     print ('zeroing depth. do not touch the gel!')
        #     if self.dm_zero_counter == 49:
        #         self.dm_zero /= self.dm_zero_counter
        # else:
        #     print ('touch me!')
        # self.dm_zero_counter += 1
        # dm = dm - self.dm_zero
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
