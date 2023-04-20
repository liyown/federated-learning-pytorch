import os
import yaml

g_configs = None

with open("config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    configs["data_config"]["partition_config"]["dir_alpha"] = 0.1
with open('config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(configs)
os.system(f"python server.py")

with open("config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    configs["data_config"]["partition_config"]["dir_alpha"] = 1
with open('config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(configs)
os.system(f"python server.py")

with open("config.yaml", "r", encoding="utf-8") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    # 修改yml文件中的参数
    configs["data_config"]["partition_config"]["dir_alpha"] = 100
with open('config.yaml', 'w', encoding="utf-8") as f:
    yaml.dump(configs, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(configs)
os.system(f"python server.py")
