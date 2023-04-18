import os

import torchvision
import seaborn as sns


import pandas as pd
import matplotlib.pyplot as plt

from dataset import CIFAR10Partitioner
from dataset.functional import partition_report

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
num_clients = 100
num_classes = 10
seed = 2021
hist_color = '#4169E1'

trainset = torchvision.datasets.CIFAR10(root="../data/", train=True, download=True)
# Hetero Dirichlet (non-iid)
# Shards (non-iid)
# Balanced IID (iid)
# Unbalanced IID (iid)
# Balanced Dirichlet (non-iid)
# Unbalanced Dirichlet (non-iid)

hetero_dir_part = CIFAR10Partitioner(trainset.targets,
                                     num_clients,
                                     balance=True,
                                     partition="iid",
                                     seed=seed)


trainset = list(trainset)[list(hetero_dir_part.client_dict[0])]


csv_file = "./cifar10_hetero_dir_0.3_100clients.csv"
partition_report(trainset.targets, hetero_dir_part.client_dict,
                 class_num=num_classes,
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file, header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
col_names = [f"class{i}" for i in range(num_classes)]
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

hetero_dir_part_df[col_names].iloc[:10].plot.barh(stacked=True)
plt.tight_layout()
plt.xlabel('sample num')
plt.savefig(f"./cifar10_hetero_dir_0.3_100clients.png", dpi=400)

clt_sample_num_df = hetero_dir_part.client_sample_count


# sns.histplot(data=clt_sample_num_df,
#              x="num_samples",
#              edgecolor='none',
#              alpha=0.7,
#              shrink=0.95,
#              color=hist_color)
# plt.savefig(f"./cifar10_hetero_dir_0.3_100clients_dist.png", dpi=400, bbox_inches='tight')
