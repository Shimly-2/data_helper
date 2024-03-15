import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from matplotlib.pyplot import MultipleLocator

json_path = "/home/jackeyjin/data_helper/datasets/subtask_datasets/different_case_bottle/force_data/GelSightL_1707331228203280_5_57.json"

with open(json_path, 'r') as file:
    json_data = json.load(file)

depth_sum = []
depth_min = []
depth_max = []
timestep = []
data = []
for idx, per_data in enumerate(json_data):
    depth_sum.append(per_data["depth_sum"])
    depth_min.append(per_data["depth_min"])
    depth_max.append(per_data["depth_max"])
    timestep.append(idx)


data.append([depth_sum])
# data.append([depth_min])
data.append([depth_max])


warnings.filterwarnings('ignore')
linestyle = ['-', '--', ':', '-.']
color = ['r', 'g', 'b', 'k']
algos = ['Depth max', 'Depth min']#, 'Depth sum']

def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    return smooth_data

# Figure大小：其实对应的是画布
# plt.subplots(2, 2, i, figsize=(7, 5))
plt.figure(figsize=(20, 4))

# algos = ["Soda Bottle"]#, "Force(DDPG)", "Force(TD3)", "Force(PPO)", "Force(SAC)"]
# data = [[] for i in range(len(algos))]
for i in range(len(algos)):
    sns.set() # 设置美化参数，一般默认就好

    y_data = smooth(data[i], 10)
    # x_data = np.arange(0, 1e6+5000, 5000)
    x_data = timestep
    # 颜色设置
    current_palette = ["#1f77b4", "#ff7f0e", "#4DAA54", '#c60f53', "#966AB9", "#8E5C41", "#817B84"] #, '#d62728'
    sns.set_palette(current_palette)

    # 应用Latex公示编辑（需要提前安装好，latex编译器并且将可执行程序添加到环境变量中）
    # plt.rc('text', usetex=True)

    # 设置Seaborn风格
    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_style({"grid.color": ".9"})
    sns.set_context("talk")  # paper， notebook， talk  poster

    # 坐标轴设置
    sns.despine(top=True, right=True, left=False, bottom=False)

    # 画图
    ax = sns.tsplot(time=x_data, data=y_data, color=current_palette[i], linestyle=linestyle[0], condition=algos[i])

    # 设置xy两轴长宽比
    # ax.set_aspect(1.2)

    # 设置坐标轴颜色以及粗细
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')

    # 设置x,y标题以及x、y两轴范围
    ax.set(xlabel="Steps", ylabel='Force Return', xlim=[0, 2800], ylim=[0, 0.3])

    # 设置x、y两轴的minor tick
    ax.xaxis.set_minor_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    # 设置图例的位置
    ax.legend(loc='upper left', frameon=False, prop={'size': 18}) #upper/lower right
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()

# plt.savefig('Soda bottle.png')
plt.show()