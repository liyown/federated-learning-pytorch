import os
import yaml

with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    print(configs)
    configs["global_config"]["record_id"] = "Cifar第二次实验 "
with open('../config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
os.system(f"python ./server.py")

with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    print(configs)
    configs["global_config"]["record_id"] = "Cifar第三次实验 "
with open('../config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
os.system(f"python ./server.py")

with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    print(configs)
    configs["global_config"]["record_id"] = "Mnist第一次实验 "
    configs["client_config"]["model"] = "MnistCNN"
    configs["data_config"]["dataset_name"] = "MNIST"
with open('../config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
os.system(f"python ./server.py")

with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    print(configs)
    configs["global_config"]["record_id"] = "Mnist第二次实验 "
with open('../config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
os.system(f"python ./server.py")

with open("../config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    print(configs)
    configs["global_config"]["record_id"] = "Mnist第三次实验 "
with open('../config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
os.system(f"python ./server.py")