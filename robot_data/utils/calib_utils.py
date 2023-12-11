import os.path as osp
import os
import json
import numpy as np


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
        vehicle_config_path,
        camera_names,
        undistort=False,
        with_hmat=False,
        gac=True,
    ):
        camera_calibrated_sensor = dict()
        for camera in camera_names:
            camera_calibrated_sensor[camera] = dict()
            config_dir = os.path.join(vehicle_config_path, camera)
            if not os.path.exists(
                    os.path.join(config_dir, f"{camera}-intrinsic.json")):
                continue
            if not os.path.exists(
                    os.path.join(config_dir,
                                 f"{camera}-to-car_center-extrinsic.json")):
                continue
            with open(os.path.join(config_dir, f"{camera}-intrinsic.json"),
                      "r") as f:
                info = json.load(f)
                # 3x 3
                key = "cam_K_new" if undistort else "cam_K"
                try:
                    intrin_mat = next(iter(
                        info.values()))["param"][key]["data"]
                    dist_mat = next(iter(
                        info.values()))["param"]["cam_dist"]["data"]
                except KeyError:
                    # TODO: remove hard-code, better way
                    if gac:
                        intrin_mat = next(iter(
                            info.values()))["param"]["cam_K"]["data"]
                        dist_mat = next(iter(
                            info.values()))["param"]["cam_dist"]["data"]
                    else:
                        continue
            with open(
                    os.path.join(config_dir,
                                 f"{camera}-to-car_center-extrinsic.json"),
                    "r") as f:
                info = json.load(f)
                # 4 x 4
                extrin_mat = next(iter(
                    info.values()))["param"]["sensor_calib"]["data"]
            camera_calibrated_sensor[camera]["intrin"] = intrin_mat
            camera_calibrated_sensor[camera]["extrin"] = extrin_mat
            if undistort:
                camera_calibrated_sensor[camera]["cam_dist"] = dist_mat
            if with_hmat:
                if not os.path.exists(
                        os.path.join(config_dir, f"{camera}-hmatrix.json")):
                    camera_calibrated_sensor[camera]["h_matrix"] = None
                    continue
                with open(os.path.join(config_dir, f"{camera}-hmatrix.json"),
                          "r") as f:
                    info = json.load(f)
                    hmat = next(iter(
                        info.values()))["param"]["h_matrix"]["data"]
                    camera_calibrated_sensor[camera]["h_matrix"] = hmat
        return camera_calibrated_sensor

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
