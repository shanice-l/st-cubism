

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    data_path = "kinetics/new_data/val_data.npy"
    label_path = "kinetics/new_data/val_label.pkl"
    data = np.load("/kinetics/new_data/val_data.npy")
    
    '''
    #transfer=2
    origin = data[:, :, :, 1, :]
    data_numpy = data - origin[:, :, :, None, :]
    data = data_numpy
    
    #transfer=1
    origin = data[:, :, :, 1, 1]
    data_numpy = data - origin[:, :, :, None, None]
    data = data_numpy
    '''

    #transfer=0
    origin = data[:, :, :, 1, 0]
    data_numpy = data- origin[:, :, :, None, None]
    data=data_numpy

    N, C, T, V, M = data.shape
    mean_map = data.mean(
        axis=2, keepdims=True).mean(
        axis=4, keepdims=True).mean(axis=0)
    # mean_map 3,1,25,1
    std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
        (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    min_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).min(axis=0)
    max_map = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).max(axis=0)
    mean_mean = (data.transpose(0, 2, 3, 4, 1).reshape(N * T * V * M, C)).mean(axis=0)
    print("self.min_map:",min_map)
    print("self.max_map:",max_map)
    print("self.mean_mean:",mean_mean)
