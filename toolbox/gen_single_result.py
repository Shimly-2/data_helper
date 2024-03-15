# -*- coding: utf-8 -*-
import time
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import sys
sys.path.append("/home/jackeyjin/data_helper/robot_data/")
# from calibration import ImageProcessor
from robot_data.utils.fast_poisson import fast_poisson
import os
from robot_data.utils.marker_detection import marker_detector
from robot_data.utils.vector_builder import vector_builder
from PIL import Image
import datetime
import copy
from tqdm import tqdm
from tqdm import trange
from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
from robot_data.utils.robot_timestamp import RobotTimestampsIncoder
from matplotlib.pyplot import MultipleLocator
import seaborn as sns

def matching_v2(test_img, ref_blur, cali, table, blur_inverse):
    """处理test_img，使其rgb归一化，方便匹配
       返回test_img每一点对应的梯度"""

    diff_temp1 = test_img - ref_blur
    diff_temp2 = diff_temp1 * blur_inverse

    diff_temp2[:, :, 0] = (diff_temp2[:, :, 0] - cali.zeropoint[0]) / cali.lookscale[0]
    diff_temp2[:, :, 1] = (diff_temp2[:, :, 1] - cali.zeropoint[1]) / cali.lookscale[1]
    diff_temp2[:, :, 2] = (diff_temp2[:, :, 2] - cali.zeropoint[2]) / cali.lookscale[2]
    diff_temp3 = np.clip(diff_temp2, 0, 0.999)
    diff = (diff_temp3 * cali.bin_num).astype(int)

    grad_img = table[diff[:, :, 0], diff[:, :, 1], diff[:, :, 2], :]

    return grad_img

def show_depth(depth, detector, disp_img):
    """绘制原始图像，原始图像与ref的差值，x和y的梯度，深度的2d和3d图"""

    scale = 3  # 3d深度图放大比例，方便观看
    origin_depth = depth
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(2, 4, 6, projection='3d')

    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    # depth=cv2.boxFilter(depth,-1,(21,21),normalize=1)
    surf = ax.plot_surface(X, Y, -depth, cmap=cm.plasma_r)
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    ax.tick_params(axis="x", labelsize=6, pad=-5)
    ax.tick_params(axis="y", labelsize=6, pad=-5)
    ax.tick_params(axis="z", labelsize=6, pad=-5)
    ax.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax.set_title('3D Depth Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    ax3 = plt.subplot(244)
    plt.title('Gradient (' + r'$\nabla_Y$' + ')', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(grad_img2[:, :, 0])

    ax2 = plt.subplot(243)
    plt.title('Gradient (' + r'$\nabla_X$' + ')', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(grad_img2[:, :, 1])

    ax4 = plt.subplot(245)
    plt.title('2D Depth Map', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(depth)

    ax1 = plt.subplot(242)
    plt.title('$img_{target}-img_{ref}$', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(cv2.cvtColor((raw_img - ref_blur) / 70, cv2.COLOR_BGR2RGB))

    ax6 = plt.subplot(241)
    plt.title('$img_{target}$', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))

    fitting = vector_builder()
    depth = depth / 3.0
    depth = fitting.rebuild_depth_with_force(depth)

    ax4 = plt.subplot(247)
    plt.title('2D Force Map', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(depth)

    ax2 = fig.add_subplot(2, 4, 8, projection='3d')
    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    # depth=cv2.boxFilter(depth,-1,(21,21),normalize=1)
    surf = ax2.plot_surface(X, Y, depth, cmap=cm.jet)
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax2.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    ax2.tick_params(axis="x", labelsize=6, pad=-5)
    ax2.tick_params(axis="y", labelsize=6, pad=-5)
    ax2.tick_params(axis="z", labelsize=6, pad=-5)
    ax2.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax2.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax2.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax2.set_title('3D Force Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    
    plt.show()

def show_depth_for_paper(depth, detector, disp_img):
    """绘制原始图像，原始图像与ref的差值，x和y的梯度，深度的2d和3d图"""

    scale = 3  # 3d深度图放大比例，方便观看
    origin_depth = copy.deepcopy(depth)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(2, 4, 6, projection='3d')

    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm

    surf = ax.plot_surface(X, Y, depth, cmap=cm.jet)
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    ax.tick_params(axis="x", labelsize=6, pad=-5)
    ax.tick_params(axis="y", labelsize=6, pad=-5)
    ax.tick_params(axis="z", labelsize=6, pad=-5)
    ax.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax.set_title('3D Depth Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    ax3 = plt.subplot(244)
    plt.title('Gradient (' + r'$\nabla_Y$' + ')', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(grad_img2[:, :, 0])

    ax2 = plt.subplot(243)
    plt.title('Gradient (' + r'$\nabla_X$' + ')', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(grad_img2[:, :, 1])

    ax4 = plt.subplot(245)
    plt.title('2D Depth Map', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(depth)

    ax1 = plt.subplot(242)
    plt.title('$img_{target}-img_{ref}$', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(cv2.cvtColor((raw_img - ref_blur) / 70, cv2.COLOR_BGR2RGB))

    ax6 = plt.subplot(241)
    plt.title('$img_{target}$', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB))

    fitting = vector_builder()
    depth = depth / 3.0
    depth = fitting.rebuild_depth_with_force(depth)

    ax4 = plt.subplot(247)
    plt.title('2D Force Map', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis('off')
    plt.imshow(depth)

    ax2 = fig.add_subplot(2, 4, 8, projection='3d')
    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm

    surf = ax2.plot_surface(X, Y, depth, cmap=cm.jet)
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax2.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    ax2.tick_params(axis="x", labelsize=6, pad=-5)
    ax2.tick_params(axis="y", labelsize=6, pad=-5)
    ax2.tick_params(axis="z", labelsize=6, pad=-5)
    ax2.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax2.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax2.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax2.set_title('3D Force Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    # Single map for paper
    img = cv2.imread('/home/jackeyjin/gelsight_cali/test/contact.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,[320,240])
    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    # In Python3 matplotlib assumes rgbdata in range 0.0 to 1.0
    img = img.astype('float32')/255
    fig2 = plt.figure(figsize=(10, 5))
    ax3 = fig2.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(0, origin_depth.shape[1], 1)
    Y = np.arange(0, origin_depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    surf2 = ax3.plot_surface(X, Y, -origin_depth, cmap=cm.jet)
    # surf2 = ax3.plot_surface(X, Y, -origin_depth, rstride=1, cstride=1, facecolors=img)
    ax3.tick_params(axis="x", labelsize=6, pad=-5)
    ax3.tick_params(axis="y", labelsize=6, pad=-5)
    ax3.tick_params(axis="z", labelsize=6, pad=-5)
    ax3.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax3.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax3.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax3.set_title('3D Depth Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    ax4 = fig2.add_subplot(1, 2, 2, projection='3d')
    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    surf2 = ax4.plot_surface(X, Y, -depth, cmap=cm.jet)
    ax4.tick_params(axis="x", labelsize=6, pad=-5)
    ax4.tick_params(axis="y", labelsize=6, pad=-5)
    ax4.tick_params(axis="z", labelsize=6, pad=-5)
    ax4.set_xlabel('Length (mm)', fontsize=6, labelpad=-8)
    ax4.set_ylabel('Width (mm)', fontsize=6, labelpad=-8)
    ax4.set_zlabel('Height (mm)', fontsize=6, labelpad=-8)
    ax4.set_title('3D Force Map', fontsize=8, pad=13)
    plt.subplots_adjust(wspace=0)

    # fig3 = plt.figure(figsize=(10, 5))
    # ax5 = fig3.add_subplot(1, 1, 1, projection='3d')
    # img = cv2.imread('test/test_img.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]
    # # In Python3 matplotlib assumes rgbdata in range 0.0 to 1.0
    # img = img.astype('float32')/255
    # # rstride，cstride相当于设置图片显示的像素
    # ax5.plot_surface(x, y, np.atleast_2d(0), rstride=1, cstride=1, facecolors=img)
    plt.show()

def plot_depth3D(depth, detector):
    """绘制原始图像，原始图像与ref的差值，x和y的梯度，深度的2d和3d图"""
    fig = plt.figure(figsize=(6, 6), dpi=300)

    sns.set_style({"grid.color": ".9"})
    sns.set_context("talk")  # paper， notebook， talk  poster
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    # depth = cv2.blur(depth, (7,7)) #数值可根据自己需要进行修改
    # depth = cv2.medianBlur(depth, 11)
    # depth=cv2.GaussianBlur(depth,(21,21),0)
    depth=cv2.boxFilter(depth,-1,(25,25),normalize=1)
    surf = ax.plot_surface(X, Y, depth, cmap=cm.plasma, label="depth") 
    '''cmap'''
    # PerceptuallyUniformSequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    # Sequential = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #               'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #               'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # Sequential2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    #                'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #                'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    # Diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #              'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    # scale = 100
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    ax.set_zlim(0, 0.25)
    ax.tick_params(axis="x", labelsize=18, pad=4)
    ax.tick_params(axis="y", labelsize=18, pad=4)
    ax.tick_params(axis="z", labelsize=18, pad=10)
    my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(np.arange(0, 26, 5))
    # plt.yticks(np.arange(0, 2, 0.3))
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    ax.set_xlabel('Length (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=10)
    ax.set_ylabel('Width (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=10)
    ax.set_zlabel('Height (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=18)

    # ax.xaxis.set_minor_locator(MultipleLocator(4))
    # ax.yaxis.set_minor_locator(MultipleLocator(4))

    # ax.set_title('3D Depth Map', fontsize=20, pad=3)
    # fig.set_size_inches(3000.0 / 100.0, 3000.0 / 100.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # ax.legend()
    # ax.legend(loc=0, frameon=False, prop={'family': 'Times New Roman', 'size': 18})
    plt.tight_layout()
    # plt.savefig("/home/jackeyjin/data_helper/datasets/subtask_datasets/A_only_test2/depth3d_297.jpg",bbox_inches='tight',pad_inches = 0.5)
    plt.show()
    # fig.canvas.draw()
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # plt.close()
    # return img

def plot_depth3D_sensor(depth, detector):
    """绘制原始图像，原始图像与ref的差值，x和y的梯度，深度的2d和3d图"""
    fig = plt.figure(figsize=(20, 20), dpi=300)

    sns.set_style({"grid.color": ".9"})
    sns.set_context("talk")  # paper， notebook， talk  poster
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = detector.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    # depth = cv2.blur(depth, (7,7)) #数值可根据自己需要进行修改
    # depth = cv2.medianBlur(depth, 11)
    # depth=cv2.GaussianBlur(depth,(21,21),0)
    # depth=cv2.boxFilter(depth,-1,(25,25),normalize=1)
    fitting = vector_builder()
    depth = depth / 3.0
    depth = fitting.rebuild_depth_with_force(depth)
    surf = ax.plot_surface(X, Y, -depth, cmap=cm.plasma_r, label="depth") 
    '''cmap'''
    # PerceptuallyUniformSequential = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    # Sequential = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #               'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #               'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    # Sequential2 = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
    #                'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
    #                'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']
    # Diverging = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    #              'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    # scale = 100
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), depth.max() - depth.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (depth.max() + depth.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z, mid_z + (max_range * 2 / scale))
    # ax.set_zlim(0, 0.25)
    ax.tick_params(axis="x", labelsize=18, pad=4)
    ax.tick_params(axis="y", labelsize=18, pad=4)
    ax.tick_params(axis="z", labelsize=18, pad=10)
    # my_x_ticks = np.arange(-5, 5, 0.5)
    # my_y_ticks = np.arange(-2, 2, 0.3)
    # plt.xticks(np.arange(0, 26, 5))
    # plt.yticks(np.arange(0, 2, 0.3))
    # labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]


    ax.set_xlabel('Length (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=10)
    ax.set_ylabel('Width (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=10)
    ax.set_zlabel('Height (mm)', fontdict={'family': 'Times New Roman', 'size': 22}, labelpad=18)

    # ax.xaxis.set_minor_locator(MultipleLocator(4))
    # ax.yaxis.set_minor_locator(MultipleLocator(4))

    # ax.set_title('3D Depth Map', fontsize=20, pad=3)
    # fig.set_size_inches(3000.0 / 100.0, 3000.0 / 100.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # ax.legend()
    # ax.legend(loc=0, frameon=False, prop={'family': 'Times New Roman', 'size': 18})
    plt.tight_layout()
    # plt.savefig("/home/jackeyjin/data_helper/datasets/subtask_datasets/force3d.jpg",bbox_inches='tight',pad_inches = 0.5)
    plt.show()
    # fig.canvas.draw()
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # plt.close()
    # return img

def make_kernal(n, k_type):
    """定义结构元素，用于后续对marker图进行膨胀操作"""
    if k_type == 'circle':
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    else:
        kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
    return kernal

def save_PC(path_PC, depth, cali):
    """保存点云"""
    f = open(path_PC, 'w')
    h = range(depth.shape[0])
    w = range(depth.shape[1])
    X = np.arange(0, depth.shape[1], 1)
    Y = np.arange(0, depth.shape[0], 1)
    X, Y = np.meshgrid(X, Y)
    Pixmm = cali.Pixmm
    X = X * Pixmm
    Y = Y * Pixmm
    for ii in h:
        for jj in w:
            tar_str = str(X[ii, jj]) + ' ' + str(Y[ii, jj]) + ' ' + str(depth[ii, jj]) + '\n'
            f.write(tar_str)
    f.close()
    return None

def tranfps(cap, fps, target_fps):
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path='/mnt/gsrobotics-Ubuntu18/examples/ros/20240206_data/20240206_GelsightR_can2_20fps.mp4'
    videoWriter = cv2.VideoWriter(video_path, fourcc, target_fps, (320,240))
    # cap = cv2.VideoCapture(video_input_path)
    cap_num = int(cap.get(7))
    idx = 0
    ret = cap.isOpened()
    assert ret, "video path not exist - {}".format(video_input_path)
    cnt = -1
    # while rval:
    video_name = video_input_path.split("/")[-1].split(".")[0]
    for i in trange(int(cap_num), colour='green', desc=f'Process---{video_name}'):
        ret, raw_img = cap.read()
        videoWriter.write(raw_img)
    videoWriter.release()


if __name__ == "__main__":
    '''用于倒置depth的config'''
    table_path = "/home/jackeyjin/Gel-Cali/cali_table/gelsight_cali_table_smooth.npy"
    ref_img_path = "/home/jackeyjin/Gel-Cali/ref_img/ref_img_update.png"
    dataset_root = "/home/jackeyjin/data_helper/datasets/subtask_datasets/A_only_test2"
    test_img_path = "/home/jackeyjin/gelsight_cali/cali_img/force_test1/gelsight_left_450.png"

    '''用于正置depth的config'''
    # ref_img_path = "/home/jackeyjin/data_helper/datasets/subtask_datasets/A_only_test2/rgb/GelSightL_1707330955058305_3_55/rgb_0.jpg"
    # test_img_path = "/home/jackeyjin/data_helper/datasets/subtask_datasets/A_only_test2/rgb/GelSightL_1707330955058305_3_55/rgb_297.jpg" # 297 745 1305 1312 1513 274 520 2591 2247 2610
    
    table = np.load(table_path)
    detector = marker_detector()
    results = []
    ref_img = cv2.imread(ref_img_path)
    marker = detector.make_mask(ref_img)
    keypoints = detector.marker_center(marker)
    marker_mask = detector.vis_mask(ref_img, keypoints)
    marker_image = np.dstack((marker_mask, marker_mask, marker_mask))
    ref_img = cv2.inpaint(ref_img, marker_mask, 3, cv2.INPAINT_TELEA)
    ref_blur = cv2.GaussianBlur(ref_img.astype(np.float32), (3, 3), 0) + 1
    blur_inverse = 1 + ((np.mean(ref_blur) / ref_blur) - 1) * 2
    red_mask = (ref_img[:, :, 2] > 12).astype(np.uint8)
    raw_img = cv2.imread(test_img_path)
    disp_img = copy.deepcopy(raw_img)

    raw_img = cv2.GaussianBlur(raw_img.astype(np.float32), (3, 3), 0)

    marker_mask = detector.make_mask(raw_img)
    kernel1 = make_kernal(25, 'circle')
    marker_mask = cv2.dilate(marker_mask, kernel1, iterations=1)  # 膨胀

    grad_img2 = matching_v2(raw_img, ref_blur, detector, table, blur_inverse)
    grad_img2[:, :, 0] = grad_img2[:, :, 0] * (1 - marker_mask) * red_mask
    grad_img2[:, :, 1] = grad_img2[:, :, 1] * (1 - marker_mask) * red_mask

    depth = fast_poisson(grad_img2[:, :, 0], grad_img2[:, :, 1])
    depth[depth < 0] = 0

    plot_depth3D(depth, detector) #正置的depth
    # plot_depth3D_sensor(depth, detector) #倒置的depth
    # show_depth(depth, detector, disp_img)
    # show_depth_for_paper(depth, detector, disp_img)