import yaml
from types import SimpleNamespace
from utils.func import get_absolute_path


# 递归将字典转换为 SimpleNamespace
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d


# 加载 YAML 文件并转换为 SimpleNamespace
def load_yaml_as_namespace(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        return dict_to_namespace(data)


def read_yaml_to_dict(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"YAML解析错误: {e}")
        return {}


def print_yaml_data(yaml_data):
    for key, value in yaml_data.items():
        print(f"{key}: {value}")


# 示例用法
if __name__ == "__main__":
    file_path = "utils/config.yaml"                             # 替换成你的YAML文件路径
    yaml_data = load_yaml_as_namespace(get_absolute_path(file_path))
    print("读取的YAML数据:")
    print_yaml_data(yaml_data)
