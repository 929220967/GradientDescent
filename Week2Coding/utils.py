import numpy as np
import h5py

def load_dataset():

    train_datasets=h5py.File('datasets/train_catvnoncat.h5','r')
    train_x_set=np.array(train_datasets["train_set_x"][:])
    train_y_set=np.array(train_datasets["train_set_y"][:])

    test_datasets=h5py.File('datasets/test_catvnoncat.h5','r')
    test_x_set=np.array(test_datasets["test_set_x"][:])
    test_y_set=np.array(test_datasets["test_set_y"][:])

    classes=np.array(test_datasets["list_classes"][:])
    train_y_set=train_y_set.reshape((1,train_y_set.shape[0]))
    test_y_set=test_y_set.reshape((1,test_y_set.shape[0]))

    return train_x_set,train_y_set,test_x_set,test_y_set,classes

