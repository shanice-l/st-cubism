# encoding: utf-8

import torch
from feeder.feeder import Feeder
import numpy as np

def fetch_dataloader(types, params):
    """
    Fetch and return train/dev
    """

    if  params.dataset_name == 'PKU_NTU_CV_temcub':
        params.train_feeder_args["data_path_source"] = params.dataset_dir+'PKU_whole_data.npy'
        params.train_feeder_args["num_frame_path_source"] = params.dataset_dir+'PKU_whole_num_frame.npy'
        params.train_feeder_args["label_path_source"] = params.dataset_dir + 'PKU_whole_label.pkl'
        params.train_feeder_args["rl_label_path_source"] = params.dataset_dir + 'PKU_whole_rl_label.pkl'#label for cubsim task

        params.train_feeder_args["data_path_target"] = params.dataset_dir + 'NTU_CV_train_data.npy'
        params.train_feeder_args["num_frame_path_target"] = params.dataset_dir + 'NTU_CV_train_num_frame.npy'
        params.train_feeder_args["rl_label_path_target"] = params.dataset_dir + 'NTU_CV_train_rl_label.pkl'#label for cubsim task

        params.test_feeder_args["data_path_source"] = params.dataset_dir + 'NTU_CV_val_data.npy'
        params.test_feeder_args["num_frame_path_source"] = params.dataset_dir  + 'NTU_CV_val_num_frame.npy'
        params.test_feeder_args["label_path_source"] = params.dataset_dir  + 'NTU_CV_val_label.pkl'
        params.test_feeder_args["rl_label_path_source"] = params.dataset_dir + 'NTU_CV_val_rl_label.pkl'#useless file

    if types == 'train':
        if not hasattr(params,'batch_size_train'):
            params.batch_size_train = params.batch_size
        #print(**params.train_feeder_args)
        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.train_feeder_args),
            batch_size=params.batch_size_train,
            shuffle=True,
            num_workers=params.num_workers,pin_memory=params.cuda)

    if types == 'test':
        if not hasattr(params,'batch_size_test'):
            params.batch_size_test = params.batch_size

        loader = torch.utils.data.DataLoader(
            dataset=Feeder(**params.test_feeder_args),
            batch_size=params.batch_size_test ,
            shuffle=False,
            num_workers=params.num_workers,pin_memory=params.cuda)

    return loader

if __name__ == '__main__':

    pass
