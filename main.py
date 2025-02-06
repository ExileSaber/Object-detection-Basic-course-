from utils.read_cfg import load_yaml_as_namespace
from utils.func import get_absolute_path
from Module.train_model import simple_model_train_F1Loss, simple_yolov3_train

if __name__ == "__main__":
    file_path = "utils/config.yaml"                             # YAML文件路径
    args = load_yaml_as_namespace(get_absolute_path(file_path))

    simple_model_train_F1Loss(args)