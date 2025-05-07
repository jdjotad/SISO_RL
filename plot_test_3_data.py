import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import glob
import seaborn as sns
import matplotlib.patches as mpatches

from utils import PlotTest

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

colors = sns.color_palette("colorblind", 7)

# Testing data
I_max = 300

test_data_folder = 'test_data'
files_Idq = glob.glob(f'{test_data_folder}/*PMSMTC*Idq*.csv')
files_Idq.sort()

# Eror in Id
fig1, ax1 = plt.subplots()
fig1.set_figheight(9)
fig1.set_figwidth(15)
ax1.set_ylabel(r'Tracking error [pu]', fontsize=fontsize)
ax1.set_xlabel(r"$\hat{i}_{d,\textrm{ref}}$ [pu]", fontsize=fontsize)

# Eror in Iq
fig2, ax2 = plt.subplots()
fig2.set_figheight(9)
fig2.set_figwidth(15)
ax2.set_ylabel(r'Tracking error [pu]', fontsize=fontsize)
ax2.set_xlabel(r"$\hat{i}_{q,\textrm{ref}}$ [pu]", fontsize=fontsize)

box_width = 0.5 + 0.1
legend_list = []
for idx, file in enumerate(files_Idq):
    color = colors[idx % 7]
    data = pd.read_csv(file, header=0)
    
    # Plot Id
    Idref = data.values[:, 0]
    Id_sorted = data.values[Idref.argsort()][:,0]
    error_Id_sorted = data.values[Idref.argsort()][:,4]

    # Negative values
    idx_mid = int(error_Id_sorted[Idref < 0].shape[0]/2)
    even = True if idx_mid*2 == error_Id_sorted[Idref < 0].shape[0] else False
    error_Id_sorted_neg = error_Id_sorted[Idref < 0].reshape(idx_mid,2) if even else error_Id_sorted[Idref < 0][:-1].reshape(idx_mid,2)
    
    # Positive values
    idx_mid = int(error_Id_sorted[Idref >= 0].shape[0]/2)
    even = True if idx_mid*2 == error_Id_sorted[Idref >= 0].shape[0] else False
    error_Id_sorted_pos = error_Id_sorted[Idref >= 0].reshape(idx_mid,2) if even else error_Id_sorted[Idref >= 0][:-1].reshape(idx_mid,2)

    data_labels_list = ["[-0.9, -0.45]", "[-0.45, 0]", "[0, 0.45]", "[0.45, 0.9]"]

    positions = [ 0 + idx*box_width,
                  5 + idx*box_width]
    
    bplot = ax1.boxplot(error_Id_sorted_neg,
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

    positions = [10 + idx*box_width,
                 15 + idx*box_width]
    bplot = ax1.boxplot(error_Id_sorted_pos,
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
    ax1.set_xticks([0 + len(files_Idq)/2*box_width, 
                   5 + len(files_Idq)/2*box_width, 
                   10 + len(files_Idq)/2*box_width, 
                   15 + len(files_Idq)/2*box_width])
        
    # Set ticks labels for x-axis
    ax1.set_xticklabels(data_labels_list, rotation='horizontal', fontsize=fontsize)

    # Plot Iq
    Iqref = data.values[:, 1]
    Iq_sorted = data.values[Iqref.argsort()][:,1]
    error_Iq_sorted = data.values[Iqref.argsort()][:,5]
   # Negative values
    idx_mid = int(error_Iq_sorted[Idref < 0].shape[0]/2)
    even = True if idx_mid*2 == error_Iq_sorted[Idref < 0].shape[0] else False
    error_Iq_sorted_neg = error_Iq_sorted[Idref < 0].reshape(idx_mid,2) if even else error_Iq_sorted[Idref < 0][:-1].reshape(idx_mid,2)
    
    # Positive values
    idx_mid = int(error_Iq_sorted[Idref >= 0].shape[0]/2)
    even = True if idx_mid*2 == error_Iq_sorted[Idref >= 0].shape[0] else False
    error_Iq_sorted_pos = error_Iq_sorted[Idref >= 0].reshape(idx_mid,2) if even else error_Iq_sorted[Idref >= 0][:-1].reshape(idx_mid,2)


    data_labels_list = ["[-0.9, -0.45]", "[-0.45, 0]", "[0, 0.45]", "[0.45, 0.9]"]

    positions = [ 0 + idx*box_width,
                  5 + idx*box_width]
    
    bplot = ax2.boxplot(error_Iq_sorted_neg,
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

    positions = [10 + idx*box_width,
                 15 + idx*box_width]
    bplot = ax2.boxplot(error_Iq_sorted_pos,
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
    ax2.set_xticks([0 + len(files_Idq)/2*box_width, 
                   5 + len(files_Idq)/2*box_width, 
                   10 + len(files_Idq)/2*box_width, 
                   15 + len(files_Idq)/2*box_width])
        
    # Set ticks labels for x-axis
    ax2.set_xticklabels(data_labels_list, rotation='horizontal', fontsize=fontsize)

    # Define legend name and color
    legend_text = " ".join(file.split("_")[file.split("_").index("PMSMTC") + 1:file.split("_").index("Idq")])
    legend_text_bold = rf"\textbf{{ {legend_text} }}"
    legend_list.append(mpatches.Patch(color=color, label=legend_text))
plt.show()

ax1.legend(handles=legend_list, loc="center left", bbox_to_anchor=(1, 0.5))
fig1.tight_layout()
ax1.grid(True, 
         axis='y',
         which='both',      # 'major', 'minor', or 'both'
         linestyle='--',    # Line style: '-', '--', '-.', ':'
         linewidth=0.5,     # Line thickness
         color='gray',     # Line color
         alpha=0.5)        # Transparency (0-1)
fig1.savefig("plots/Test_error_PMSMTC_Idref.pdf")

ax2.legend(handles=legend_list, loc="center left", bbox_to_anchor=(1, 0.5))
fig2.tight_layout()
ax2.grid(True, 
         axis='y',
         which='both',      # 'major', 'minor', or 'both'
         linestyle='--',    # Line style: '-', '--', '-.', ':'
         linewidth=0.5,     # Line thickness
         color='gray',     # Line color
         alpha=0.5)        # Transparency (0-1)
fig2.savefig("plots/Test_error_PMSMTC_Iqref.pdf")