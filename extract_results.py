import pandas as pd
import numpy as np
import os
import sys
from glob import glob
from tkinter import filedialog 
from tkinter import * 

def getDirPath():
    root = Tk() 
    root.withdraw() 
    folder_selected = filedialog.askdirectory()
    if len(folder_selected) != 0:
        return folder_selected
    else:
        print("Error: any folder was not selected!")
        return -1

def getSubDirs(mainDirPath):
    list_subfolders_with_paths = [f.path for f in os.scandir(mainDirPath) if f.is_dir()]
    # print(len(list_subfolders_with_paths))
    return list_subfolders_with_paths

def getAllFiles(dirPath, fileExt):
    all_csv_files = []
    for path, subdir, files in os.walk(dirPath):
        for file in glob(os.path.join(path, fileExt)):
            all_csv_files.append(file)
    return all_csv_files

def get_immediate_subdirectories(mainDir):
    return [name for name in os.listdir(mainDir)
            if os.path.isdir(os.path.join(mainDir, name))]

def get_file_name_and_ext(filePath):
    file_name, file_ext = os.path.splitext(os.path.basename(filePath))
    return file_name, file_ext

def read_csv(csvFilePath):
    csv_file_name, csv_file_ext = get_file_name_and_ext(csvFilePath)    
    df_csv = pd.read_csv(csvFilePath)
    print(f"The file name:{csv_file_name}\nThe content:\n\t{df_csv.to_string()}")
    return df_csv

def parse_df_content(dataFrame):
    # print(f"Dataframe columns:\n\t{dataFrame.columns}")
    df_cols = dataFrame.columns
    first_col_values = dataFrame[dataFrame.columns[0]]
    # print(f"Values of the first column:\n{first_col_values}")

    new_df_col_names = []
    new_df_col_values = []
    for dfcs in df_cols[1:]:
        for fcv1_2_ID, fcv1_2 in enumerate(first_col_values[:2]):
            new_df_col_names.append(f"{dfcs}({fcv1_2})")
            new_df_col_values.append(dataFrame.loc[fcv1_2_ID, dfcs])
    
    new_df_col_names.append(df_cols[4])
    new_df_col_values.append(dataFrame.loc[3, df_cols[4]])
    
    new_df_col_names.append(first_col_values[3])
    new_df_col_values.append(dataFrame.loc[3, df_cols[3]])
    
    for fcv7_9_ID, fcv7_9 in enumerate(first_col_values[7:]):
        new_df_col_names.append(fcv7_9)
        new_df_col_values.append(dataFrame.loc[fcv7_9_ID+7, df_cols[1]])
    # print(f"Column names of the main table:\n\t{new_df_col_names}")
    # print(f"Column values of the main table:\n\t{new_df_col_values}")
    return [new_df_col_names, new_df_col_values]


if __name__ == "__main__":
    main_dir_path = getDirPath()

    dataset_dir_paths = getSubDirs(mainDirPath=main_dir_path)
    print(f"\nThe selected directory:\n{main_dir_path}\n\nSubdirectories:\n{dataset_dir_paths}\n")

    dataset_names = get_immediate_subdirectories(mainDir=main_dir_path)
    print(f"The list of immediate subdirectory names:\n{dataset_names}\n")

    # The ds_res_path variable indicates to the results of a particular dataset.
    for drpID, ds_res_path in enumerate(dataset_dir_paths):
        # Identificator whether to write column names or not.
        # We only need to write column names for each dataset results on time.
        # This identificator prevents us to write column names again and again.
        iden_col_names = 0

        dataset_main_results = []
        
        dataset_name = dataset_names[drpID]
        
        # class_dir_path - a path to the results of a particular classifier. 
        classifier_dir_names = get_immediate_subdirectories(ds_res_path)
        print(f"\nSubdir names:\n\t{classifier_dir_names}")
        
        class_dir_paths = getSubDirs(ds_res_path)
        print(f"\nThe path by dataset results:\n\t{ds_res_path}\n\tThe paths by classifiers:\n\t{class_dir_paths[1:-1]}")
        
        # cdp variable keeps the classifier directory path
        for cdpID, cdp in enumerate(class_dir_paths):
            if cdpID == 0 or cdpID == len(class_dir_paths)-1:
                continue
            else:
                classifier_name = classifier_dir_names[cdpID]
                print(f"Dataset:{dataset_name}\nClassifier:{classifier_name}")
                
                full_features_file_path = f"{cdp}/full_features.csv"
                ff_df_content = read_csv(full_features_file_path)
                # Parsed df column names and values for full features
                ff_parsed_df_cn, ff_parsed_df_cv = parse_df_content(dataFrame=ff_df_content)                
                
                if iden_col_names == 0:
                    ff_parsed_df_cn.insert(0,"Results")
                    dataset_main_results.append(ff_parsed_df_cn)
                    ff_parsed_df_cv.insert(0,f"{classifier_name}_FF")
                    dataset_main_results.append(ff_parsed_df_cv)
                else:
                    ff_parsed_df_cv.insert(0,f"{classifier_name}_FF")
                    dataset_main_results.append(ff_parsed_df_cv)
                
                print(f"Column names of the main table:\n\t{ff_parsed_df_cn}")
                print(f"Column values of the main table:\n\t{ff_parsed_df_cv}\n")
                
                iden_col_names += 1

                # regressor_dir_path - a path to the results of a particular regressor. 
                regressor_dir_names = get_immediate_subdirectories(cdp)
                print(f"Subdir names:{regressor_dir_names}")
                regressor_dir_paths = getSubDirs(cdp)
                print(f"Subdir paths:{regressor_dir_paths}")

                for rdpID, rdp in enumerate(regressor_dir_paths):
                    regressor_name = regressor_dir_names[rdpID]
                    # The path list of all csv files that keep the feature selection results, 
                    # which are in the classifiers' folders.
                    all_csv_files_paths_4_fs_res = getAllFiles(rdp, '*.csv')
                    print(f"Feature selection result (csv) files for {regressor_name}:\n\t{all_csv_files_paths_4_fs_res}\
                          \nThe number of files: {len(all_csv_files_paths_4_fs_res)}\n")
                    # Get feature selection results by first, feature selection method;
                    # second, regressor; third, classifier; then last, dataset.
                    for fs_csv_path in all_csv_files_paths_4_fs_res:
                        fs_name, fs_file_ext = get_file_name_and_ext(fs_csv_path)
                        
                        fs_df_content = read_csv(fs_csv_path)
                        fs_parsed_df_cn, fs_parsed_df_cv = parse_df_content(dataFrame=fs_df_content)     
                        print(f"Column names of the main table:\n\t{fs_parsed_df_cn}")
                        print(f"Column values of the main table:\n\t{fs_parsed_df_cv}\n")
                        
                        if fs_name == "backward_sequential_based_fs":
                           fs_name = 'BS'
                        elif fs_name == "correlation_coefficient_based_fs":
                           fs_name = 'CC'
                        elif fs_name == "forward_sequential_based_fs":
                           fs_name = 'FS'
                        elif fs_name == 'importance_based_fs':
                           fs_name = 'IP'                
                        fs_parsed_df_cv.insert(0,f"{classifier_name}_{regressor_name}_{fs_name}")
                        dataset_main_results.append(fs_parsed_df_cv)
        
        print(f"\nMain results:\n{dataset_main_results}\n")
        df_dataset_main_results = pd.DataFrame(dataset_main_results[1:], columns=dataset_main_results[0])
        # df_dataset_main_results.to_excel(f"{ds_res_path}/{dataset_name}.xlsx", index=False)
        df_dataset_main_results.to_csv(f"{ds_res_path}/{dataset_name}.csv", index=False)



