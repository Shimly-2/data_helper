import logging
import os.path as osp
from robot_data.utils.utils import read_json, write_json, read_txt
import os
from abc import ABC, abstractmethod
from multiprocessing_logging import install_mp_handler
from multiprocessing import Pool, Lock, Value
import time

logging.basicConfig(
    level=logging.INFO,
    format=
    "'%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'",
)

lock = Lock()
counter = Value('i', 0)
total_time = Value('f', 0)
total = Value('i', 0)


class BaseDataProcessor(ABC):

    def __init__(
        self,
        workspace,
        case_names_file=None,
        meta_save_name="raw_data.json",
        task_infos_save_name="task_infos.json",
        log_file_dir=None,
        update_meta_file=True,
        update_task_infos_file=True,
        manual_meta_setting: dict = {},
        manual_task_infos_setting: dict = {},
        multiprocess_pool=1,
    ):
        self.workspace = workspace
        self.pool = multiprocess_pool
        self.update_meta_file = update_meta_file
        self.update_task_infos_file = update_task_infos_file
        self.logger = self.get_logger()
        self.manual_meta_setting = manual_meta_setting
        self.manual_task_infos_setting = manual_task_infos_setting
        self.meta_save_name = meta_save_name
        self.task_infos_save_name = task_infos_save_name
        self.case_names_file = case_names_file

        if log_file_dir:
            if not osp.exists(log_file_dir):
                os.makedirs(log_file_dir)

            self.logger = self.get_logger(log_file_path=osp.join(
                log_file_dir, f"{self.__class__.__name__}.log"))
        else:
            self.logger = self.get_logger()

    @abstractmethod
    def process(self, meta, task_infos):
        raise NotImplementedError

    def get_offline_meta(self):
        if not osp.exists(osp.join(self.workspace, self.meta_save_name)):
            self.logger.warn(f"cant find {self.meta_save_name}")
            if not self.case_names_file or not osp.exists(
                    osp.join(self.workspace, self.case_names_file)):
                self.logger.warn(
                    "cant find case_names file, return empty meta")
                return {}
            else:
                case_names = read_txt(
                    osp.join(self.workspace, self.case_names_file))
                return {
                    case_name: {
                        'case_name': case_name
                    }
                    for case_name in case_names
                }
        else:
            meta = read_json(osp.join(self.workspace, self.meta_save_name))
            meta = {m["case_name"]: m for m in meta}
            return meta

    def get_offline_task_infos(self):
        if not osp.exists(osp.join(self.workspace, self.task_infos_save_name)):
            self.logger.warn(f"cant find {self.task_infos_save_name}")
            return {}
        else:
            task_infos = read_json(
                osp.join(self.workspace, self.task_infos_save_name))[0]
            return task_infos

    def get_logger(self, log_file_path=None):
        logger = logging.getLogger(self.__class__.__name__)
        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )

        if log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(level=logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if self.pool > 1:
            install_mp_handler()

        return logger

    def sub_func(self, func, *args, **kwargs):
        global total, counter, total_time, lock
        start_time = time.time()
        res = func(*args, **kwargs)
        pid = os.getpid()
        end_time = time.time()
        with lock:  # type: ignore
            counter.value += 1  # type: ignore
            total_time.value += (end_time - start_time)  # type: ignore
        eta_time = (total_time.value / counter.value) * (  # type: ignore
            total.value - counter.value) / 60  # type: ignore
        self.logger.info(
            f"Process : {pid} : finish multiprocess task [{counter.value}] / [{total.value}], ETA: {int(eta_time)} minutes"  # type: ignore
        )
        return res

    def multiprocess_run(self, func, args_list):
        global total, counter, total_time, lock
        if self.pool <= 1:
            self.logger.warn(
                'define pool num large than 1 if using multiprocess_run')
            return []
        pool = Pool(self.pool)
        total.value = len(args_list)
        counter.value = 0
        total_time.value = 0.0
        self.logger.info(f'start multiprocessing, task nums : {total.value}')

        results = []
        for args in args_list:
            results.append(pool.apply_async(self.sub_func, args=(func, *args)))

        pool.close()
        pool.join()

        results = [res.get() for res in results]

        return results

    def __call__(self, meta=None, task_infos=None):
        if not meta:
            meta = self.get_offline_meta()
        if not task_infos:
            task_infos = self.get_offline_task_infos()

        if self.manual_meta_setting:
            if self.manual_meta_setting.get("set_before_process", False):
                for _, v in meta.items():
                    v.update(self.manual_meta_setting)

        if self.manual_task_infos_setting:
            if self.manual_task_infos_setting.get("set_before_process", False):
                task_infos.update(self.manual_task_infos_setting)

        meta, task_infos = self.process(meta=meta, task_infos=task_infos)

        if self.manual_meta_setting:
            if not self.manual_meta_setting.get("set_before_process", False):
                for _, v in meta.items():
                    v.update(self.manual_meta_setting)

        if self.manual_task_infos_setting:
            if not self.manual_task_infos_setting.get("set_before_process",
                                                      False):
                task_infos.update(self.manual_task_infos_setting)

        if self.update_meta_file:
            autolabel_meta_path = osp.join(self.workspace, self.meta_save_name)
            for case_name, m in meta.items():
                if "case_name" not in m:
                    m["case_name"] = case_name
            write_json(list(meta.values()), autolabel_meta_path)
        if self.update_task_infos_file:
            task_infos_path = osp.join(self.workspace,
                                       self.task_infos_save_name)
            write_json(task_infos, task_infos_path)
