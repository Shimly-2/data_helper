import yaml
from abc import ABC, abstractclassmethod
from robot_data.utils.registry_factory import DATA_PROCESSER_REGISTRY
import argparse


class BaseRunner(ABC):
    def __init__(self, config_file) -> None:
        super().__init__()
        with open(config_file) as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.processors = self.build_processors()

    def build_processors(self):
        processors = []
        for processor_config in self.config["data_processor"]:
            processor = DATA_PROCESSER_REGISTRY[processor_config["type"]]
            processors.append(processor(**processor_config["kwargs"]))
        return processors

    def run(self):
        meta = []
        task_infos = {}
        for processor in self.processors:
            results = processor(meta, task_infos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str)
    # parser.add_argument("--CPU_PER_TASK", type=int)

    args = parser.parse_args()

    runner = BaseRunner(config_file=args.config_path)
    runner.run()
