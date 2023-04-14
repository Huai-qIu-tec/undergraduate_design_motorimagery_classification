import numpy as np
import torch
import pickle

train_data_dir = '../数据集/BCICIV_2a_gdf_resample512/A02T_train_data_resample512.npz'
train_label_dir = '../数据集/BCICIV_2a_gdf_resample512/A02T_train_label_resample512.npz'
test_data_dir = '../数据集/BCICIV_2a_gdf_resample512/A02T_test_data_resample512.npz'
test_label_dir = '../数据集/BCICIV_2a_gdf_resample512/A02T_test_label_resample512.npz'

train_data = np.load(train_data_dir)
train_label = np.load(train_label_dir)

test_data = np.load(test_data_dir)
test_label = np.load(test_label_dir)

data = np.vstack([train_data['arr_0'], test_data['arr_0']])
label = np.hstack([train_label['arr_0'], test_label['arr_0']])

data_left_class = torch.from_numpy(data[label == 0])
data_right_class = torch.from_numpy(data[label == 1])
data_foot_class = torch.from_numpy(data[label == 2])
data_tongue_class = torch.from_numpy(data[label == 3])

torch.save(data_left_class, '../数据集/BCICIV_2a_4class_data/left_data.pt')
torch.save(data_right_class, '../数据集/BCICIV_2a_4class_data/right_data.pt')
torch.save(data_foot_class, '../数据集/BCICIV_2a_4class_data/foot_data.pt')
torch.save(data_tongue_class, '../数据集/BCICIV_2a_4class_data/tongue_data.pt')