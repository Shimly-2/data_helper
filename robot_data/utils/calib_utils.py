import os.path as osp
import os
import json
import numpy as np
import cv2

class calib_loader():

    def __init__(self) -> None:
        pass

    def gen_cam_h_by_case(self, car_center_dir):
        with open(
                osp.join(car_center_dir,
                         "car_center-to-ground_projection-extrinsic.json"),
                "r",
        ) as f:
            info = json.load(f)
            mat = next(iter(info.values()))["param"]["sensor_calib"]["data"]
            mat = np.array(mat)
            cam_h = float(mat[2][3]) * -1.0
            tmp_mat = np.eye(4)
            tmp_mat[2][3] = -1.0 * cam_h
            assert np.all(tmp_mat == mat), f"{tmp_mat} vs {mat}"
        assert cam_h is not None
        return cam_h

    def _load_camera_calibrated_sensor(
        self,
        sensor_config_path,
        sensor_names,
        undistort=False,
    ):
        '''
            camera_calibrated_sensor:
                gelsight_left:
                    intrin:
                    cam_dist:
                gelsight_right:
                    intrin:
                    cam_dist:
        '''
        camera_calibrated_sensor = dict()
        for sensor in sensor_names:
            camera_calibrated_sensor[sensor] = dict()
            config_dir = os.path.join(sensor_config_path, sensor)
            if not os.path.exists(
                    os.path.join(config_dir, f"{sensor}-intrinsic.json")):
                continue
            with open(os.path.join(config_dir, f"{sensor}-intrinsic.json"),
                      "r") as f:
                info = json.load(f)
                # 3x 3
                key = "cam_K"
                intrin_mat = next(iter(
                    info.values()))["param"][key]["data"]
                dist_mat = next(iter(
                    info.values()))["param"]["cam_dist"]["data"]
            camera_calibrated_sensor[sensor]["intrin"] = intrin_mat
            if undistort:
                camera_calibrated_sensor[sensor]["cam_dist"] = dist_mat
        return camera_calibrated_sensor
    
    def gen_calib_results(
        self, 
        imgs,
        camera_calibrated_sensor,
        sensor_names
    ):
        '''
            imgs:
                gelsight_left:
                    img_path:
                    img_origin:
                gelsight_right:
                    img_path:
                    img_origin:
            dst:
                gelsight_left:
                    img_distort:
                gelsight_right:
                    img_distort:
        '''
        for sensor in sensor_names:
            dst[sensor] = dict()
            intrin = camera_calibrated_sensor[sensor]["intrin"]
            dist = camera_calibrated_sensor[sensor]["cam_dist"]
            img = imgs[sensor]
            # img_distort = cv2.undistort(img, np.array(intrin), np.array(dist))
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrin, dist, (w,h), 0, (w,h))
            dst[sensor]["img_distort"] = cv2.undistort(img, intrin, dist, None, newcameramtx)

            # 剪裁图像
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
        return dst

    def gen_calib_sensor_by_case(self, cams, calib_dir, GAC=True):
        car_center_dir = osp.join(calib_dir, "car_center")

        # if GAC:
        #     calib_dir = osp.join(calib_dir, "conf")
        calib_sensor = self._load_camera_calibrated_sensor(calib_dir,
                                                           cams,
                                                           undistort=True,
                                                           gac=GAC)
        cam_h = self.gen_cam_h_by_case(car_center_dir)
        return calib_sensor, cam_h


def project_to_camera(points, camera_intrinsic, sensor_extrinsic):
    points = points[..., points[2, :] > -2]
    # sensor to camera, 4xN
    camera_points = sensor_extrinsic.dot(np.array(points))
    # z is front in camera, 3xN
    camera_points = camera_points[:-1]
    # camera to image, 3xN
    img_points = camera_intrinsic.dot(camera_points)
    img_points = img_points[..., img_points[2, :] >= 0]  # 只保留深度为正的点
    img_points[0, :] /= img_points[2, :]
    img_points[1, :] /= img_points[2, :]
    img_points = img_points.transpose()  # Nx3 array
    return img_points
