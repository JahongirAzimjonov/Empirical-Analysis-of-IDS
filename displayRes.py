import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
from glob import glob
from tkinter import filedialog 
from tkinter import * 
import tkinter as tk
from scipy.interpolate import make_interp_spline, BSpline

def display_species():
    species = ("Adelie", "Chinstrap", "Gentoo")
    penguin_means = {
        'Bill Depth': (18.35, 18.43, 14.98),
        'Bill Length': (38.79, 48.83, 47.50),
        'Flipper Length': (189.95, 195.82, 217.19),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 250)

    plt.show()

def extractingImportantMetrics(df, importantMetrics):
    df_temp = df[importantMetrics]
    return df_temp

def determine_facecolor(ds_name):
    if ds_name == "nbaiot21":
        return "whitesmoke"
    elif ds_name == "nslkdd21":
        return "mistyrose"
    elif ds_name == "insdn20":
        return "thistle"
    elif ds_name == "cicids17":
        return "wheat"
    elif ds_name == "botiot18":
        return "lightcyan"
    else:
        return "lightpink"

def determine_acc_bar_and_plot_colors(ds_name):
    if ds_name == "nbaiot21":
        return ("lightcoral", "red", "o")
    elif ds_name == "nslkdd21":
        return ("y", "darkorange", "s")
    elif ds_name == "insdn20":
        return ("palegreen", "darkgreen", "*")
    elif ds_name == "cicids17":
        return ("lightskyblue", "teal", "^")
    elif ds_name == "botiot18":
        return ("cornflowerblue", "navy", "p")
    else:
        return ("violet", "deeppink", "d")

def display_train_test_time_performance(df, fn):
    dataset_name = fn
    # species = ("Adelie", "Chinstrap", "Gentoo")

    # time_performance_metrics = [ 'Training Time', 'Test Time-All Packets', 'Test Time-Per Packet']
    time_performance_metrics = ['MethodShortName', 'Training Time', 'Test Time-All Packets']
    df_temp = extractingImportantMetrics(df, time_performance_metrics)

    tp_metric_types = df_temp[time_performance_metrics[0]]
    tp_metrics_with_values = {}
    for tpmet in time_performance_metrics[1:-1]:
        tp_metrics_with_values[tpmet] = df_temp[tpmet]
    # penguin_means = {
    #     'Bill Depth': (18.35, 18.43, 14.98),
    #     'Bill Length': (38.79, 48.83, 47.50),
    #     'Flipper Length': (189.95, 195.82, 217.19),
    # }

    x = np.arange(len(tp_metric_types))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(21,11), layout='constrained')

    bar_color, plot_color, marker = determine_acc_bar_and_plot_colors(dataset_name)
    
    # colors = ["magenta", "royalblue"]
    # colors = ["c", "mediumblue", "yellow", "orange", "springgreen", "green"]
    colors = [bar_color]
    color_ID = 0
    for attribute, measurement in tp_metrics_with_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_ID], alpha=0.51)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
        color_ID += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    ax.set_title(f"The Train & Test Metrics of IDSs, Trained on \n the Full and Subsets of the '{dataset_name.upper()}' dataset", \
                 fontsize=29, fontweight="bold")
    ax.set_xlabel(f"IDSs by Classifiers, Kernel Regressors and Feature Selectors", \
                  fontsize=25, fontweight="bold")
    ax.set_ylabel("Packet Processing Time (milliseconds)", fontsize=25, fontweight="bold")
    
    facecolor = determine_facecolor(dataset_name)
    ax.set_facecolor(facecolor)
    
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, tp_metric_types, \
                  rotation=45, ha='right', fontsize=17, fontweight="bold")

    y_max = max(df_temp['Training Time'])
    y_min = min(df_temp['Training Time'])
    delta_y = 2 * (y_max - y_min) // len(df_temp['Training Time'])
    plt.yticks(ticks=np.arange(0.0, y_max, delta_y), fontsize=15, fontweight="bold", rotation=30)
    ax.set_ylim(0, y_max+delta_y)    

    legend_properties = {'weight':'bold', 'size':19}
    # plt.legend(title="$\\bf{Features\ Selected\ with:}$", \
    #            title_fontsize=21, prop=legend_properties)
    
    if dataset_name == "nslkdd21":
        ax.legend(loc='upper left', ncols=len(time_performance_metrics)-1,\
              prop=legend_properties, fancybox=True, framealpha=1)
    else:
        ax.legend(loc='upper right', ncols=len(time_performance_metrics)-1,\
              prop=legend_properties, fancybox=True, framealpha=1)

    x = df[time_performance_metrics[0]] # 0 - MethodShortName
    y_train = df['Training Time']
    # y_test_all = df['Test Time-All Packets']
    # addlabels(x=x, y=y_train, ha="center", va="center", rotation=30)
    # addlabels(x=x, y=y_test_all, ha="center", va="center", rotation=30)
    # bar_color, plot_color, marker = determine_acc_bar_and_plot_colors(dataset_name)
    plt.plot(x, y_train+0.5, color=plot_color, linewidth=3, linestyle="-", marker=marker, markersize=13, alpha=.9)
    # plt.plot(x, y_test_all+0.5, color=colors[1], linewidth=3, linestyle="-", marker=marker, markersize=13, alpha=.9)

    plt.savefig(f"./TimePerformMetrics-{dataset_name}.png", dpi = 150)
    plt.show()

def display_entire_metrics(df, fn):
    dataset_name = fn
    # species = ("Adelie", "Chinstrap", "Gentoo")

    # time_performance_metrics = [ 'Training Time', 'Test Time-All Packets', 'Test Time-Per Packet']
    accuracy_metrics = ['MethodShortName', 'Precision(Normal)', 'Precision(Attack)', \
                    'Re-Call(Normal)', 'Re-Call(Attack)', 'F1-Score(Normal)', 'F1-Score(Attack)', 'Accuracy']
    df_temp = extractingImportantMetrics(df, accuracy_metrics)

    acc_metric_types = df_temp[accuracy_metrics[0]]
    acc_metrics_with_values = {}
    for acmet in accuracy_metrics[1:-1]:
        acc_metrics_with_values[acmet] = df_temp[acmet]
    # penguin_means = {
    #     'Bill Depth': (18.35, 18.43, 14.98),
    #     'Bill Length': (38.79, 48.83, 47.50),
    #     'Flipper Length': (189.95, 195.82, 217.19),
    # }

    x = np.arange(len(acc_metric_types))  # the label locations
    width = 0.123  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(29,17), layout='constrained')

    # colors = ["orange", "lightgreen", "royalblue", "brown", "magenta", "teal"]
    colors = ["c", "mediumblue", "yellow", "orange", "springgreen", "green"]
    color_ID = 0
    for attribute, measurement in acc_metrics_with_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[color_ID])
        # ax.bar_label(rects, padding=3)
        multiplier += 1
        color_ID += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    
    ax.set_title(f"The Evaluation Metrics of IDSs, Trained on the Full and Subsets of the '{dataset_name.upper()}' dataset", \
                 fontsize=29, fontweight="bold")
    ax.set_xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", \
                  fontsize=25, fontweight="bold")
    ax.set_ylabel("IDSs Metrics by Percentage", fontsize=25, fontweight="bold")
    
    facecolor = determine_facecolor(dataset_name)
    ax.set_facecolor(facecolor)
    
    # ax.set_ylabel('Length (mm)')
    # ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width, acc_metric_types, \
                  rotation=45, ha='right', fontsize=17, fontweight="bold")
    legend_properties = {'weight':'bold', 'size':19}
    # plt.legend(title="$\\bf{Features\ Selected\ with:}$", \
    #            title_fontsize=21, prop=legend_properties)
    ax.legend(loc='upper center', ncols=len(accuracy_metrics)-2,\
              prop=legend_properties, fancybox=True, framealpha=1)
    plt.yticks(ticks=np.arange(0.0, 1.05, 0.05), fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.1)

    plt.savefig(f"./EvalMetricsc-{dataset_name}.png", dpi = 150)
    plt.show()

def get_file_path(initDirPath):
        file_path = ""
        file_select_window = tk.Tk()
        file_select_window.withdraw()
        # print(tk1)
        # show an "Open" dialog box and return the path to the 	selected file
        filePath = filedialog.askopenfilename(initialdir = initDirPath,
                                            title = "Select file",
                                            filetypes = (("image files","*.csv"),
                                                        ("image files", "*.xls"),
                                                        ("image files", "*.xlsx"),
                                                        ("all files","*.*")))
        file_select_window.destroy()

        if (filePath!=""):
            file_path = filePath
            return file_path
        else:
            print("Any file has not been selected!")
            return -1

def get_file_name_and_ext(file_path):
    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    return file_name, file_ext

def get_folder_path():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

def display_entire_results(df, fn):
    dataset_name = fn.upper()

    # time_performance_metrics = [ 'Training Time', 'Test Time-All Packets', 'Test Time-Per Packet']
    accuracy_metrics = ['MethodShortName', 'Precision(Normal)', 'Precision(Attack)', \
                    'Re-Call(Normal)', 'Re-Call(Attack)', 'F1-Score(Normal)', 'F1-Score(Attack)', 'Accuracy']
    
    df_temp = df[accuracy_metrics]
    print(f"df cols:\n{df.columns}\nJust accuracy metrics:\n{accuracy_metrics}\n")

    # plot grouped bar chart
    precision_cols = ['MethodShortName', 'Precision(Normal)', 'Precision(Attack)']
    re_call_cols = ['MethodShortName', 'Re-Call(Normal)', 'Re-Call(Attack)']
    f1_score_cols = ['MethodShortName', 'F1-Score(Normal)', 'F1-Score(Attack)']

    df_temp[precision_cols].plot(x='MethodShortName',
            kind='bar',
            stacked=False,
            title='Detection Accuracy Evaluation Results')
    # plt.figure(figsize=(23,15), layout="constrained")
    plt.rcParams["figure.figsize"] = 23,15
    plt.xticks(rotation=45, ha='right', fontsize=17, fontweight="bold")
    plt.yticks(ticks=np.arange(0.0, 1.01, 0.05), fontsize=17, fontweight="bold")
    
    plt.ylim(0.0, 1.01)
    
    plt.title(f"The Accuracy of IDSs, Trained on the Full and Subsets of the '{dataset_name}' dataset", fontsize=29, fontweight="bold")
    plt.xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", fontsize=23, fontweight="bold")
    plt.ylabel("IDSs Accuracy", fontsize=23, fontweight="bold")
    plt.show()

def display_entire_classification_acc_metrics(df, fn):
    dataset_name = fn

    bar_width = 0.2

    plt.figure(figsize=(53,15), layout="constrained")
    plt.xticks(rotation=45, ha='right', fontsize=17, fontweight="bold")
    plt.yticks(ticks=np.arange(0.0, 1.03, 0.05), fontsize=17, fontweight="bold")
    
    plt.ylim(0.0, 1.01)
    
    plt.title(f"The Accuracy of IDSs, Trained on the Full and Subsets of the '{dataset_name.upper()}' dataset", fontsize=29, fontweight="bold")
    plt.xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", fontsize=23, fontweight="bold")
    plt.ylabel("IDSs Accuracy", fontsize=23, fontweight="bold")
    
    # # Making a smooth curve
    # x_temp = np.linspace(0, len(x), len(x))
    # print(f"The length of x_temp:{len(x_temp)}")
    # xnew = np.linspace(x_temp.min(), x_temp.max(), 300) 
    # #define spline
    # spl = make_interp_spline(x_temp, y, k=3)
    # y_smooth = spl(xnew)
    # plt.plot(xnew,y_smooth) 
    
    # time_performance_metrics = [ 'Training Time', 'Test Time-All Packets', 'Test Time-Per Packet']
    accuracy_metrics = ['MethodShortName', 'Precision(Normal)', 'Precision(Attack)', \
                    'Re-Call(Normal)', 'Re-Call(Attack)', 'F1-Score(Normal)', 'F1-Score(Attack)', 'Accuracy']
    
    df_temp = df[accuracy_metrics]
    # print(f"df cols:\n{df.columns}\nJust accuracy metrics:\n{accuracy_metrics}\n")
    x_bar = np.arange(len(df['MethodShortName']))
    print(f"x_bar:{x_bar}\n")
    # plt.bar(x=x_bar, height=df_temp['Accuracy'], color="#C20078", width=bar_width)
    plt.plot(df_temp['MethodShortName'], df_temp['Accuracy'], marker='o', linestyle='-', color="magenta", linewidth=2, markersize=11)


    # plot grouped bar chart
    precision_cols = ['Precision(Normal)', 'Precision(Attack)']
    re_call_cols = ['Re-Call(Normal)', 'Re-Call(Attack)']
    f1_score_cols = ['F1-Score(Normal)', 'F1-Score(Attack)']
    
    plt.bar(x=x_bar-0.9, height=df_temp[precision_cols[0]], color="lightgreen", width=bar_width)
    azure = "#069AF3"
    # plt.plot(df_temp['MethodShortName'], df_temp[precision_cols[0]], marker='*', linestyle='-', color=azure, linewidth=1, markersize=9)
    blue = "#0343DF"
    plt.bar(x=x_bar-0.6, height=df_temp[precision_cols[1]], color="green", width=bar_width)
    # plt.plot(df_temp['MethodShortName'], df_temp[precision_cols[1]], marker='o', linestyle='-.', color=blue, linewidth=1, markersize=9)

    # gold = "#FFD700"
    gold = "#DBB40C"
    plt.bar(x=x_bar-0.3, height=df_temp[re_call_cols[0]], color=gold, width=bar_width)
    aqua = "#13EAC9"
    # plt.plot(df_temp['MethodShortName'], df_temp[re_call_cols[0]], marker='^', linestyle='-', color=aqua, linewidth=1, markersize=9)
    # gold = "#DBB40C"
    plt.bar(x=x_bar+0.3, height=df_temp[re_call_cols[1]], color=gold, width=bar_width)
    # plt.plot(df_temp['MethodShortName'], df_temp[re_call_cols[1]], marker='+', linestyle='-.', color="aquamarine", linewidth=1, markersize=9)

    violet = "#EE82EE"
    plt.bar(x=x_bar+0.6, height=df_temp[f1_score_cols[0]], color=violet, width=bar_width)
    aqua = "#13EAC9"
    # plt.plot(df_temp['MethodShortName'], df_temp[re_call_cols[0]], marker='^', linestyle='-', color=aqua, linewidth=1, markersize=9)
    violet = "#9A0EEA"
    plt.bar(x=x_bar+0.9, height=df_temp[f1_score_cols[1]], color=violet, width=bar_width)
    # plt.plot(df_temp['MethodShortName'], df_temp[re_call_cols[1]], marker='+', linestyle='-.', color="aquamarine", linewidth=1, markersize=9)

    # plt.grid(axis = 'y')

    plt.show()

# function to add value labels
def addlabels(x,y,ha,va,rotation):
    for i in range(len(x)):
        plt.text(x=i, y=y[i]-0.035, s=f"{y[i]}", ha=ha, va=va, rotation=rotation, fontdict={"size":15, "weight":"bold"})

# Display classifier accuracy by a trained dataset
def display_signle_classification_acc_metrics(df, fn):
    dataset_name = fn

    plt.figure(figsize=(23,15), layout="constrained")
    plt.xticks(rotation=45, ha='right', fontsize=17, fontweight="bold")
    plt.yticks(ticks=np.arange(0.0, 1.01, 0.05), fontsize=17, fontweight="bold")
    
    plt.ylim(0.0, 1.05)
    
    plt.title(f"The Accuracy of IDSs, Trained on the Full and Subsets of the '{dataset_name.upper()}' dataset",\
               fontsize=29, fontweight="bold")
    plt.xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", fontsize=23, fontweight="bold")
    plt.ylabel("IDSs Accuracy", fontsize=23, fontweight="bold")

    cols = df.columns
    print(f"Columns: {cols}")
    i = 0

    x = df[cols[0]] # 0 - MethodShortName
    y = df['Accuracy']
    addlabels(x=x, y=y, ha="center", va="center", rotation=51)
    
    # # Making a smooth curve
    # x_temp = np.linspace(0, len(x), len(x))
    # print(f"The length of x_temp:{len(x_temp)}")
    # xnew = np.linspace(x_temp.min(), x_temp.max(), 300) 
    # #define spline
    # spl = make_interp_spline(x_temp, y, k=3)
    # y_smooth = spl(xnew)
    # plt.plot(xnew,y_smooth)

    bar_color, plot_color, marker = determine_acc_bar_and_plot_colors(dataset_name)
    plt.plot(x, y+0.005, color=plot_color, linewidth=3, linestyle="-", marker=marker, markersize=13, alpha=.7)
    plt.bar(height=y, x=x, color=bar_color, width=0.7, alpha=0.5)    

    # plt.grid(axis = 'y')
    plt.savefig(f"./Accuracy-{dataset_name}.png", dpi = 150)
    plt.show()

def display_training_time_metrics(df, fn):
    dataset_name = fn

    plt.figure(figsize=(23,15), layout="constrained")
    plt.xticks(rotation=45, ha='right', fontsize=17, fontweight="bold")
    # plt.yticks(ticks=np.arange(0.0, 1.01, 0.05), fontsize=17, fontweight="bold")
    
    # plt.ylim(0.0, 1.05)
    
    plt.title(f"Training Time of IDSs, Trained on the Full and Subsets of the '{dataset_name.upper()}' dataset",\
               fontsize=29, fontweight="bold")
    plt.xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", fontsize=23, fontweight="bold")
    plt.ylabel("IDSs Train Time per Packet in Milliseconds", fontsize=23, fontweight="bold")

    cols = df.columns
    print(f"Columns: {cols}")
    i = 0

    x = df[cols[0]] # 0 - MethodShortName
    y = df['Training Time']
    y_max = max(df['Training Time'])
    y_min = min(df['Training Time'])
    delta_y = (y_max - y_min) / len(df['Test Time-Per Packet'])
    
    plt.yticks(ticks=np.arange(0.0, y_max, 2*delta_y), fontsize=15, fontweight="bold", rotation=30)
    plt.ylim(0-y_min/2, y_max+delta_y)
    # addlabels(x=x, y=y-delta_y, ha="center", va="center", rotation=51)
    for i in range(len(x)):
        # print(f"delta_y:{delta_y}\ty[{i}]:{y[i]}\n")
        if y[i] <= 2*delta_y:
            plt.text(x=i, y=y[i]+delta_y, s=f"{round(y[i],1)}", ha="center", va="center", rotation=90, fontdict={"size":15, "weight":"bold"})
        else:
            plt.text(x=i, y=y[i]/2, s=f"{round(y[i],1)}", ha="center", va="center", rotation=90, fontdict={"size":15, "weight":"bold"})

    # bar_color, plot_color, marker = determine_acc_bar_and_plot_colors(dataset_name)
    bar_color, plot_color, marker = "darkred", "darkred", "o"
    if dataset_name == "nbaiot21":
        bar_color, plot_color, marker = ("darkred", "darkred", "o")
    elif dataset_name == "nslkdd21":
        bar_color, plot_color, marker = ("darkolivegreen", "darkolivegreen", "s")
    elif dataset_name == "insdn20":
        bar_color, plot_color, marker = ("indigo", "indigo", "*")
    elif dataset_name == "cicids17":
        bar_color, plot_color, marker = ("midnightblue", "midnightblue", "^")
    elif dataset_name == "botiot18":
        bar_color, plot_color, marker = ("teal", "teal", "p")
    else:
        bar_color, plot_color, marker = ("violet", "violet", "d")

    plt.plot(x, y, color=plot_color, linewidth=3, linestyle="-", marker=marker, markersize=13, alpha=.71)
    plt.bar(height=y, x=x, color=bar_color, width=0.7, alpha=0.31)    

    # plt.grid(axis = 'y')
    plt.savefig(f"./TrainingTime-{dataset_name}.png", dpi = 150)
    plt.show()

def display_signle_packet_test_time_metrics(df, fn):
    dataset_name = fn

    plt.figure(figsize=(23,15), layout="constrained")
    plt.xticks(rotation=45, ha='right', fontsize=17, fontweight="bold")
    # plt.yticks(ticks=np.arange(0.0, 1.01, 0.05), fontsize=17, fontweight="bold")
    
    # plt.ylim(0.0, 1.05)
    
    plt.title(f"Test Time of IDSs, Trained on the Full and Subsets of the '{dataset_name.upper()}' dataset",\
               fontsize=29, fontweight="bold")
    plt.xlabel(f"IDSs by the Classifiers and Regression Model-based Feature Selectors", fontsize=23, fontweight="bold")
    plt.ylabel("IDSs Test Time per Packet in Milliseconds", fontsize=23, fontweight="bold")

    cols = df.columns
    print(f"Columns: {cols}")
    i = 0

    x = df[cols[0]] # 0 - MethodShortName
    y = df['Test Time-Per Packet']
    y_max = max(df['Test Time-Per Packet'])
    y_min = min(df['Test Time-Per Packet'])
    delta_y = (y_max - y_min) / len(df['Test Time-Per Packet'])
    
    plt.yticks(ticks=np.arange(0.0, y_max, 2*delta_y), fontsize=15, fontweight="bold", rotation=30)
    plt.ylim(0-17*y_min, y_max+delta_y)
    # addlabels(x=x, y=y-delta_y, ha="center", va="center", rotation=51)
    for i in range(len(x)):
        # print(f"delta_y:{delta_y}\ty[{i}]:{y[i]}\n")
        if y[i] <= 2*delta_y:
            plt.text(x=i, y=y[i]+3*delta_y, s=f"{y[i]}", ha="center", va="center", rotation=90, fontdict={"size":15, "weight":"bold"})
        else:
            plt.text(x=i, y=y[i]/2, s=f"{round(y[i],7)}", ha="center", va="center", rotation=90, fontdict={"size":15, "weight":"bold"})

    # bar_color, plot_color, marker = determine_acc_bar_and_plot_colors(dataset_name)
    bar_color, plot_color, marker = "darkorange", "darkorange", "o"
    if dataset_name == "nbaiot21":
        bar_color, plot_color, marker = ("darkorange", "darkorange", "o")
    elif dataset_name == "nslkdd21":
        bar_color, plot_color, marker = ("green", "green", "s")
    elif dataset_name == "insdn20":
        bar_color, plot_color, marker = ("purple", "purple", "*")
    elif dataset_name == "cicids17":
        bar_color, plot_color, marker = ("deepskyblue", "deepskyblue", "^")
    elif dataset_name == "botiot18":
        bar_color, plot_color, marker = ("brown", "brown", "p")
    else:
        bar_color, plot_color, marker = ("deeppink", "deeppink", "d")
    plt.plot(x, y, color=plot_color, linewidth=3, linestyle="-", marker=marker, markersize=13, alpha=.71)
    plt.bar(height=y, x=x, color=bar_color, width=0.7, alpha=0.31)    

    # plt.grid(axis = 'y')
    plt.savefig(f"./TestTimePerPacket-{dataset_name}.png", dpi = 150)
    plt.show()

def displayComparativeResultsByDatasets(mainDirPath):
    # print(f"Main Directory Path: {mainDirPath}\n")
    subdirs = [x[1] for x in os.walk(mainDirPath)][0]

    csv_path = f"{mainDirPath}/{subdirs[0]}/{subdirs[0]}.csv"    
    csv_df = pd.read_csv(csv_path)    
    ids_methods = csv_df[csv_df.columns[0]]    
    general_acc_results = {"IDS_Methods":ids_methods}

    paths_to_acc_res_by_datasets = []
    for subdir in subdirs:
        csv_file_path = f"{mainDirPath}/{subdir}/{subdir}.csv"
        paths_to_acc_res_by_datasets.append(csv_file_path)
        
        csv_file_df = pd.read_csv(csv_file_path)
        
        # ids_method_name_column = csv_file_df[csv_file_df.columns[0]]
        ids_accuracy = csv_file_df["Accuracy"]
        general_acc_results[subdir] = ids_accuracy
        # print(f"IDS method names:\n{ids_method_name_column}\nIDS accuracy:\n{ids_accuracy}")
    # print(f"The paths to the '.csv' result files by datasets:\n{paths_to_acc_res_by_datasets}\n")
    print(f"General accuracy results by datasets: {general_acc_results}\n")
    gen_acc_df = pd.DataFrame(general_acc_results)
    gen_acc_df.to_csv("./GenAccRes.csv", index=True)



if __name__ == "__main__":
    # path_to_eval_metrics_results = get_file_path("./")
    # file_name, file_ext = get_file_name_and_ext(path_to_eval_metrics_results)
    # df_emr = pd.read_csv(path_to_eval_metrics_results)

    # # display_species()
    
    # # Displaying the Precision, Re-Call and F1-Score values of the evaluation results.
    # display_entire_metrics(df_emr, file_name)
    
    # # display_train_test_time_performance(df_emr, file_name)
    
    # # Displaying the training time values of the evaluation results.
    # display_training_time_metrics(df_emr, file_name)
    
    # # Displaying the test time per packet values of the evaluation results.
    # display_signle_packet_test_time_metrics(df_emr, file_name)
    
    # # display_entire_results(df=df_emr, fn=file_name)
    
    # # display_entire_classification_acc_metrics(df=df_emr, fn=file_name)

    # # Displaying the accuracy values of the evaluation results.
    # display_signle_classification_acc_metrics(df_emr, file_name)

    selectedDirPath = get_folder_path()
    displayComparativeResultsByDatasets(selectedDirPath)



