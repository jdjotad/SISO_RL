import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import glob
import seaborn as sns
import matplotlib.patches as mpatches

from utils import PlotTest

# Create file with data
data_file = open("test_data.txt", "w")
data_file.write("   Mean     Median      Q1      Q3     Low_whisker     Up_whisker\n")

plt.ion()
sns.set_palette("colorblind")  # Seaborn's colorblind-friendly palette
plt.rc('text', usetex=True)  # This requires a LaTeX installation
plt.rc('font', family='serif')
fontsize = 31
plt.rcParams.update({
                    "font.size": fontsize,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    })

# colors = plt.get_cmap("tab10").colors  # 10 distinct colors
colors = sns.color_palette("colorblind", 7)

# Training data
train_data_folder = 'train_data'
files_train = glob.glob(f'{train_data_folder}/*.csv')
files_train.sort()

fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(15)
# ax.set_ylabel(r'\textbf{Average reward}')
# ax.set_xlabel(r'\textbf{Episode}')
ax.set_ylabel(r'Average reward')
ax.set_xlabel(r'Episode')
legend_list = []
for idx, file in enumerate(files_train):
    color = colors[idx % 7]
    data = pd.read_csv(file, header=0)
    data = data.values.flatten()
    ax.plot(data)
    
    filename = file.split('.csv')[0]
    legend_text = " ".join(filename.split("_")[filename.split("_").index("PMSM") + 1:])
    legend_list.append(mpatches.Patch(color=color, label=legend_text))
ax.legend(handles=legend_list, loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True, 
         which='both',      # 'major', 'minor', or 'both'
         linestyle='--',    # Line style: '-', '--', '-.', ':'
         linewidth=0.5,     # Line thickness
         color='gray',     # Line color
         alpha=0.5)        # Transparency (0-1)
plt.tight_layout()
fig.savefig("Train_average_reward.pdf")

# Testing data
I_max = 300

test_data_folder = 'test_data'
files_Id = glob.glob(f'{test_data_folder}/*Id*.csv')
files_Iq = glob.glob(f'{test_data_folder}/*Iq*.csv')
files_Id.sort()
files_Iq.sort()

# Eror in Id
fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(15)
# ax.set_ylabel(r'\textbf{Tracking error [pu]}', fontsize=fontsize)
# ax.set_xlabel(r"$\mathbf{\hat{i}_{d,\textbf{\textrm{ref}}}}$ \textbf{[pu]}", fontsize=fontsize)
ax.set_ylabel(r'Tracking error [pu]', fontsize=fontsize)
ax.set_xlabel(r"$\hat{i}_{d,\textrm{ref}}$ [pu]", fontsize=fontsize)

box_width = 0.5 + 0.1
legend_list = []
for idx, file in enumerate(files_Id):
    color = colors[idx % 7]
    data = pd.read_csv(file, header=0)
    data_values = data.values.transpose().reshape(8000*5,4)

    data_name = (data.keys().to_list()[0]).split(" = ")[0]
    data_labels = [float(header.split(" = ")[-1]) for header in data.keys().to_list()]
    # I was here!
    data_labels = np.linspace(-0.5, 0.5, 5)
    # data_labels_list = [f"[{I_max*data_left:.2f}, {I_max*data_right:.2f}]" for data_left, data_right in zip(data_labels, np.roll(data_labels, -1))][:-1]
    data_labels_list = [f"[{data_left:.2f}, {data_right:.2f}]" for data_left, data_right in zip(data_labels, np.roll(data_labels, -1))][:-1]

    positions = [ 0 + idx*box_width,
                  5 + idx*box_width,
                 10 + idx*box_width,
                 15 + idx*box_width]
    
    bplot = ax.boxplot(data_values,
                    positions=positions,
                    patch_artist=True,  # fill with color
                    meanline=True,
                    showmeans=True,
                    showfliers=False,
                    medianprops={'linewidth': 1.5},
                    meanprops={'linewidth': 1.5})
    
    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set_color('black')        

    # Set number of ticks for x-axis
    ax.set_xticks([0 + len(files_Id)/2*box_width, 
                   5 + len(files_Id)/2*box_width, 
                   10 + len(files_Id)/2*box_width, 
                   15 + len(files_Id)/2*box_width])
    legend_text = " ".join(file.split("_")[file.split("_").index("PMSM") + 1:file.split("_").index("Id")])
    legend_text_bold = rf"\textbf{{ {legend_text} }}"
    legend_list.append(mpatches.Patch(color=color, label=legend_text))
    
    boxplot_data = cbook.boxplot_stats(data_values)
    for idx in range(data_values.shape[1]):
        data_file.write(f"{legend_text} Id_ref = {data_labels_list[idx]} {boxplot_data[idx]['mean']:.4f} & " + 
                                                           f"{boxplot_data[idx]['med']:.4f} & " +
                                                           f"{boxplot_data[idx]['q1']:.4f} & " +
                                                           f"{boxplot_data[idx]['q3']:.4f} & " +
                                                           f"{boxplot_data[idx]['whislo']:.4f} & " +
                                                           f"{boxplot_data[idx]['whishi']:.4f}\n")
        
    # Set ticks labels for x-axis
    ax.set_xticklabels(data_labels_list, rotation='horizontal', fontsize=fontsize)
    plt.show()

data_file.write("\n\n")

ax.legend(handles=legend_list, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid(True, 
         axis='y',
         which='both',      # 'major', 'minor', or 'both'
         linestyle='--',    # Line style: '-', '--', '-.', ':'
         linewidth=0.5,     # Line thickness
         color='gray',     # Line color
         alpha=0.5)        # Transparency (0-1)
fig.savefig("Test_error_Idref.pdf")

# Error in Iq
fig, ax = plt.subplots()
fig.set_figheight(9)
fig.set_figwidth(15)
ax.set_ylabel('Tracking error [pu]', fontsize=fontsize)
ax.set_xlabel(r"$\hat{i}_{q,\textrm{ref}}$ [pu]", fontsize=fontsize)

box_width = 0.5 + 0.1
legend_list = []
for idx, file in enumerate(files_Iq):
    color = colors[idx % 7]
    data = pd.read_csv(file, header=0)
    data_values = data.values.transpose().reshape(8000*5,4)

    data_name = (data.keys().to_list()[0]).split(" = ")[0]
    data_labels = [float(header.split(" = ")[-1]) for header in data.keys().to_list()]

    data_labels = np.linspace(-0.5, 0.5, 5)
    # data_labels_list = [f"[{I_max*data_left:.2f}, {I_max*data_right:.2f}]" for data_left, data_right in zip(data_labels, np.roll(data_labels, -1))][:-1]
    data_labels_list = [f"[{data_left:.2f}, {data_right:.2f}]" for data_left, data_right in zip(data_labels, np.roll(data_labels, -1))][:-1]


    positions = [ 0 + idx*box_width,
                  5 + idx*box_width,
                 10 + idx*box_width,
                 15 + idx*box_width]
    
    bplot = ax.boxplot(data_values,
                    positions=positions,
                    patch_artist=True,  # fill with color
                    meanline=True,
                    showmeans=True,
                    showfliers=False,
                    medianprops={'linewidth': 1.5},
                    meanprops={'linewidth': 1.5})
    # fill with colors
    for patch in bplot['boxes']:
        patch.set_facecolor(color)        
    for median in bplot['medians']:
        median.set_color('black')   

    # Set number of ticks for x-axis
    ax.set_xticks([0 + len(files_Iq)/2*box_width, 
                   5 + len(files_Iq)/2*box_width, 
                   10 + len(files_Iq)/2*box_width, 
                   15 + len(files_Iq)/2*box_width])
    legend_text = " ".join(file.split("_")[file.split("_").index("PMSM") + 1:file.split("_").index("Iq")])
    legend_list.append(mpatches.Patch(color=color, label=legend_text))
    
    boxplot_data = cbook.boxplot_stats(data_values)
    for idx in range(data_values.shape[1]):
        data_file.write(f"{legend_text} Iq_ref = {data_labels_list[idx]} {boxplot_data[idx]['mean']:.4f} & " + 
                                                           f"{boxplot_data[idx]['med']:.4f} & " +
                                                           f"{boxplot_data[idx]['q1']:.4f} & " +
                                                           f"{boxplot_data[idx]['q3']:.4f} & " +
                                                           f"{boxplot_data[idx]['whislo']:.4f} & " +
                                                           f"{boxplot_data[idx]['whishi']:.4f}\n")
        
    # Set ticks labels for x-axis
    ax.set_xticklabels(data_labels_list, rotation='horizontal', fontsize=fontsize)
    plt.show()

ax.legend(handles=legend_list, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.grid(True, 
         axis='y',
         which='both',      # 'major', 'minor', or 'both'
         linestyle='--',    # Line style: '-', '--', '-.', ':'
         linewidth=0.5,     # Line thickness
         color='gray',     # Line color
         alpha=0.5)        # Transparency (0-1)
fig.savefig("Test_error_Iqref.pdf")

data_file.close()