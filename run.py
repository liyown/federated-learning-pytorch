# 运行fedavg中的main.py文件
import os
import fedavg.main
import fedtransfermer.main
# 运行fedtransfermer中的main.py文件
path = "fedtransfermer/"
os.chdir(path)
fedtransfermer.main.run()
# 运行fedavg中的main.py文件
path = "fedavg/"
os.chdir(path)
fedavg.main.run()


