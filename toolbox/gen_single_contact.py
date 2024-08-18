from robot_data.data_processor.BaseDataProcessor import BaseDataProcessor
import cv2
import time
import os
from tqdm import tqdm
import math
import multiprocessing
import os.path as osp
import numpy as np
import copy
from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
from robot_data.utils.robot_timestamp import RobotTimestampsIncoder
from robot_data.apis.reconstruction3d import Reconstruction3D
from robot_data.apis.finger import Finger, RGB2NormNetR1, RGB2NormNetR15
from robot_data.configs import setting
from robot_data.apis import marker_detection
from robot_data.apis import find_marker
from robot_data.utils.calib_utils import calib_loader
from robot_data.utils.image_reader import ImageReader
from robot_data.apis.flow_calculation import flow_calculate_global, estimate_uv
from robot_data.apis.contact_detection import contact_detection_v2
import json

# @DATA_PROCESSER_REGISTRY.register("Tactile_Learner")
class tactile_learner(BaseDataProcessor):

    def __init__(
        self,
        workspace,
        tactile_root,
        description_save_root,
        dataset_root,
        sensor_config_path,
        sensor_name,
        model_file_path,
        gpuorcpu="cpu",
        use_video=False,
        use_calibrate=False,
        use_undistort=True,
        border_size=25,
        is_save_video=False,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)
        self.tactile_root = tactile_root
        self.description_save_root = description_save_root
        self.dataset_root = dataset_root
        self.sensor_config_path = sensor_config_path
        self.sensor_name = sensor_name
        self.use_undistort = use_undistort
        self.model_file_path = model_file_path
        self.gpuorcpu = gpuorcpu
        self.use_video = use_video
        self.use_calibrate = use_calibrate

        setting.init()
        # self.new_gt_save_dir = new_gt_save_dir
        # os.makedirs(self.describe_save_root, exist_ok=True)

        self.timestamp_maker = RobotTimestampsIncoder()
        # os.makedirs(new_gt_save_dir, exist_ok=True)
    
    def init_nn(self, net_path, gpuorcpu):
        self.nn = Reconstruction3D(Finger.R15)
        self.net = self.nn.load_nn(net_path, gpuorcpu)
    
    def init_flow_matcher(self, ref_img):
        mask = marker_detection.find_marker(ref_img)
        ### find marker centers
        mc = marker_detection.marker_center(mask, ref_img)
        mc_sorted1 = mc[mc[:,0].argsort()]
        mc1 = mc_sorted1[:setting.N_]
        mc1 = mc1[mc1[:,1].argsort()]

        mc_sorted2 = mc[mc[:,1].argsort()]
        mc2 = mc_sorted2[:setting.M_]
        mc2 = mc2[mc2[:,0].argsort()]
        N_= setting.N_
        M_= setting.M_
        fps_ = setting.fps_
        x0_ = np.round(mc1[0][0])
        y0_ = np.round(mc1[0][1])
        dx_ = mc2[1, 0] - mc2[0, 0]
        dy_ = mc1[1, 1] - mc1[0, 1]
        self.flow_matcher = find_marker.Matching(N_,M_,fps_,x0_,y0_,dx_,dy_)
        self.radius ,self.coverage = self.compute_tracker_gel_stats(mask)

    def init_cailb(self, sensor_config_path, sensor_names, undistort):
        self.calibrator = calib_loader()
        self.calib_param = self.calibrator._load_camera_calibrated_sensor(sensor_config_path, sensor_names, undistort=undistort,)

    def compute_tracker_gel_stats(self, thresh):
        numcircles = setting.numcircles
        mmpp = setting.mmpp
        true_radius_mm = setting.true_radius_mm
        true_radius_pixels = true_radius_mm / mmpp
        circles = np.where(thresh)[0].shape[0]
        circlearea = circles / numcircles
        radius = np.sqrt(circlearea / np.pi)
        radius_in_mm = radius * mmpp
        percent_coverage = circlearea / (np.pi * (true_radius_pixels) ** 2)
        return radius_in_mm, percent_coverage*100.
    
    def gen_imgs_list(self):
        imgs_list_generator = ImageReader("path","RGB")
        self.imgs_list = imgs_list_generator.prepare_imgs(self.tactile_root)

    def gen_per_optical_flow_description(self, raw_img_list):
        if not raw_img_list[0].endswith(".jpg"):
            ref_img = cv2.imread(raw_img_list[0] + ".jpg")
        else:
            ref_img = cv2.imread(raw_img_list[0])
        self.init_nn(self.model_file_path, self.gpuorcpu)
        self.init_flow_matcher(ref_img)
        self.init_cailb(self.sensor_config_path, self.sensor_name, self.use_undistort)
        contact_path = os.path.dirname(raw_img_list[0]).replace("rgb","contact_test")
        # os.makedirs(contact_path, exist_ok=True)
        # if not force_data_path.endswith(".json"):
        #     force_data_path = force_data_path + ".json"
        # object_name = flow_path.split("/")[-3]
        # view = flow_path.split("/")[-1]
        force_data = []
        for i in range(1):
            per_force_data = dict()
            # if not raw_img_path.endswith(".jpg"):
            #     raw_img_path = raw_img_path + ".jpg"
            imgs = dict()
            imgs["gelsight"] = dict()
            imgs["gelsight"]["img_origin"] = cv2.imread(raw_img_list[1])
            final_listn = [[0]*9 for _ in range(7)]
            final_list_num = 0
            x1, y1, x2, y2, u, v = [],[],[],[],[],[]
            slip_monitor = {}
            previous_u_sum = np.array([0])
            previous_v_sum = np.array([0])
            if self.use_calibrate:
                frame = self.calibrator.gen_calib_results(imgs, self.camera_calibrated_sensor, self.sensor_names)
            
            #TODO
            for sensor in self.sensor_name:
                imgs[sensor]["img_depth"] = self.nn.get_depthmap(self.net, imgs[sensor]["img_origin"], setting.MASK_MARKERS_FLAG)

                ### find marker masks
                imgs[sensor]["img_mask"] = marker_detection.find_marker(imgs[sensor]["img_origin"])

                ### find marker centers
                imgs[sensor]["img_mc"] = marker_detection.marker_center(imgs[sensor]["img_mask"], imgs[sensor]["img_origin"])
            if self.use_calibrate == False:
                ### matching init
                # self.flow_matcher.init(imgs["gelsight"]["img_mc"])

                # ### matching
                # self.flow_matcher.run()

                # ### matching result
                # """
                # output: (Ox, Oy, Cx, Cy, Occupied) = flow
                #     Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
                #     Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
                #     Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                #         e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
                # """
                # flow = self.flow_matcher.get_flow()

                # # calculate flow
                # x1, y1, x2, y2, u, v = flow_calculate_global(flow)
                # u_sum = np.array(u)
                # v_sum = np.array(v)
                # x2_center = np.expand_dims(np.array(x2),axis = 1)
                # y2_center = np.expand_dims(np.array(y2),axis = 1)
                # x1_center = np.expand_dims(np.array(x1),axis = 1)
                # y1_center = np.expand_dims(np.array(y1),axis = 1)
                # p2_center = np.expand_dims(np.concatenate((x2_center,y2_center),axis = 1),axis = 0)
                # p1_center = np.expand_dims(np.concatenate((x1_center,y1_center),axis = 1),axis = 0)

                # tran_matrix, _ = cv2.estimateAffinePartial2D(p1_center,p2_center,False)

                # if tran_matrix is not None:
                #     u_estimate, v_estimate = estimate_uv(tran_matrix, x1, y1, u_sum, v_sum, x2, y2)
                #     vel_diff = np.sqrt((u_estimate - u_sum)**2 + (v_estimate - v_sum)**2)
                #     u_diff = u_estimate - u_sum
                #     v_diff = v_estimate - v_sum
                # if np.abs(np.mean(v_sum)) > np.abs(np.mean(u_sum)) + 2:
                #     thre_slip_dis = 3.5
                # else:
                #     thre_slip_dis = 4.5

                #     numofslip = np.sum(vel_diff > thre_slip_dis)
    
                #     abs_change_u = np.abs(previous_u_sum - np.mean(u_sum))
                #     abs_change_v = np.abs(previous_v_sum - np.mean(v_sum))
                #     abs_change = np.sqrt(abs_change_u**2+abs_change_v**2)
                #     # diff_img_sum = np.sum(np.abs(previous_image.astype(np.int16) - final_image.astype(np.int16)))
                    
                #     thre_slip_num = 7
                #     slip_indicator = numofslip > thre_slip_num
                #     static_flag = False

                # if tran_matrix is None:
                #     slip_monitor['values'] = [np.mean(np.array(u)),np.mean(np.array(v)),np.arcsin(self.tran_matrix[1,0])/np.pi*180]
                # else:
                #     slip_monitor['values'] = [np.mean(np.array(u)),np.mean(np.array(v)),0.]

                # slip_monitor['name'] = str(slip_indicator)

                # previous_slip = slip_indicator
                # previous_u_sum = np.mean(np.array(u))
                # previous_v_sum = np.mean(np.array(v))

                
                # if slip_indicator: 
                #     print("slip!") 
                #     slip_indicator = False

                # contact detection####################################################
                for sensor in self.sensor_name:
                    imgs[sensor]["img_contact"] = contact_detection_v2(imgs[sensor]["img_origin"], ref_img)
                    imgs[sensor]["img_contact"] = cv2.medianBlur(imgs[sensor]["img_contact"],25)
                    contours = cv2.findContours(imgs[sensor]["img_contact"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    fill_image = np.zeros((240, 320),dtype=np.uint8) 

                    if contours is not None:
                        area = []
                        topk_contours =[]
                        x, y = imgs[sensor]["img_contact"].shape
                        for i in range(len(contours)):
                            # 对每一个连通域使用一个掩码模板计算非0像素(即连通域像素个数)
                            single_masks = np.zeros((x, y),dtype=np.uint8) 
                            fill_image = cv2.fillConvexPoly(single_masks, np.array(contours[i],dtype = np.int), 255)
                            pixels = cv2.countNonZero(fill_image)
                            if(pixels<200):
                                fill_image = cv2.fillConvexPoly(single_masks, contours[i], 0)
                            else:
                                area.append(pixels)
                            imgs[sensor]["img_contact"] = cv2.bitwise_or(single_masks,fill_image)
                        # 画接触区域轮廓线####################################################
                        if len(area): 
                            cv2.drawContours(imgs[sensor]["img_contact"], contours, -1, (0,255,0), 2)
                        bitand = cv2.medianBlur(imgs[sensor]["img_contact"],21)
                        if(len(imgs[sensor]["img_contact"][imgs[sensor]["img_contact"]==255])<20):
                            imgs[sensor]["img_origin"][:,:,1] = imgs[sensor]["img_origin"][:,:,1]
                        else:
                            imgs[sensor]["img_origin"][:,:,1] = imgs[sensor]["img_origin"][:,:,1] + bitand/7
                        ####################################################################
                    ####################################################################
                        
                        # calculate final contact list
                        # Ox, Oy, Cx, Cy, Occupied = flow
                        # for i in range(len(Ox)):
                        #     for j in range(len(Ox[i])):
                        #         if(fill_image[int(Cy[i][j])][int(Cx[i][j])]==255):
                        #             final_listn[i][j] = 1

                        # imgs[sensor]["img_flow"] = copy.deepcopy(imgs[sensor]["img_origin"])
                        # imgs[sensor]["img_uv_flow"] = copy.deepcopy(imgs[sensor]["img_origin"])
                        # marker_detection.draw_flow(imgs[sensor]["img_flow"], flow)
                        # marker_detection.draw_flow_uv(imgs[sensor]["img_uv_flow"], flow, u_estimate, v_estimate, final_listn)
                        
                        # cv2.imwrite(raw_img_path.replace("rgb","contact_test"),imgs[sensor]["img_origin"])
                        cv2.imshow("contact",imgs[sensor]["img_origin"])
                        cv2.waitKey(0)
                        # self.logger.info(f"Successsfully saved {save_path}")
                        # marker_detection.draw_flow_contact(frame, flow, final_listn)
                        # marker_detection.draw_flow_uv_contact(frameuv, flow, u_estimate, v_estimate, final_listn)
        
        # self.logger.info(f'Done: {flow_path}')
    
    def gen_meta_infos(self, meta):
        video_paths = []
        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    video_paths.append(os.path.join(dirpath, filename))
        for video_path in video_paths:  #依次读取视频文件
            '''/root/raw_meta/xxx.mp4-->/root/rgb/xxx/'''
            save_new_root = video_path.replace("raw_meta", "rgb").split(".")[0]
            meta.append({"raw_meta_root": save_new_root})
        return meta
    
    def process(self, meta, task_infos):
        if isinstance(meta, list):
            if len(meta) == 0:
                meta = self.gen_meta_infos(meta)
        results = []
        if self.pool > 1:
            args_list = []
            for per_meta in meta:
                raw_img_list = os.listdir(per_meta["raw_meta_root"]) 
                '''sort rgb_1.jpg/xxx_1.jpg'''
                raw_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1])) 
                raw_img_list = [os.path.join(per_meta["raw_meta_root"], x) for x in raw_img_list]
                flow_path = os.path.dirname(raw_img_list[0]).replace("rgb", "flow")
                contact_path = os.path.dirname(raw_img_list[0]).replace("rgb","contact")
                uv_flow_path = os.path.dirname(raw_img_list[0]).replace("rgb","uv_flow")
                force_data_path = os.path.dirname(raw_img_list[0]).replace("rgb", "force_uv_data")
                args_list.append((raw_img_list, flow_path, contact_path, uv_flow_path, force_data_path))
            results = self.multiprocess_run(self.gen_per_optical_flow_description, args_list)
        else:
            for idx, per_meta in enumerate(meta):
                raw_img_list = os.listdir(per_meta["raw_meta_root"]) 
                '''sort rgb_1.jpg/xxx_1.jpg'''
                raw_img_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1])) 
                raw_img_list = [os.path.join(per_meta["raw_meta_root"], x) for x in raw_img_list]
                flow_path = os.path.dirname(raw_img_list[0]).replace("rgb", "flow")
                contact_path = os.path.dirname(raw_img_list[0]).replace("rgb","contact_test")
                uv_flow_path = os.path.dirname(raw_img_list[0]).replace("rgb","uv_flow")
                force_data_path = os.path.dirname(raw_img_list[0]).replace("rgb", "force_uv_data")
                self.logger.info(
                    f"Start process {idx+1}/{len(meta)}")
                results.append(
                    self.gen_per_optical_flow_description(raw_img_list, flow_path, contact_path, uv_flow_path, force_data_path))
        return meta, task_infos
    
if __name__ == "__main__":
    tactile = tactile_learner(
        workspace=None,
        tactile_root="/mnt/gsrobotics-Ubuntu18/examples/ros/20240206_data/light_auto_sel/rgb/GelSightL_1714047492181235",
        description_save_root=None,
        dataset_root="/mnt/gsrobotics-Ubuntu18/examples/ros/20240206_data/light_auto_sel",
        sensor_config_path="/home/jackeyjin/data_helper/datasets/demo_calibration/gelsight_left",
        sensor_name=["gelsight"],
        model_file_path="/home/jackeyjin/data_helper/robot_data/configs/nnmini.pt",
        gpuorcpu="cpu",
        use_video=False,
        use_calibrate=False,
        use_undistort=True,
        border_size=25,
        is_save_video=False,
    )
    img_list = [
        "/mnt/gsrobotics-Ubuntu18/examples/ros/20240206_data/light_auto_sel/rgb/GelSightL_1714047492181235/rgb_0.jpg",
        "/mnt/gsrobotics-Ubuntu18/examples/ros/20240206_data/light_auto_sel/rgb/GelSightL_1714047492181235/rgb_137.jpg"
    ]
    tactile.gen_per_optical_flow_description(img_list)