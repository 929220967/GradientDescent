import numpy as np
import h5py

def load_dataset():
    train_dataset=h5py.File('datasets/train_catvnoncat.h5','r')  #若是要向文件中写入数据就要用w,r为读取数据
    train_set_x_orig=np.array(train_dataset["train_set_x"][:]) #保存的是训练集里面的图像数据（209张64*64的图像）
    train_set_y_orig=np.array(train_dataset["train_set_y"][:]) #保存的是训练集的图像对应的分类值(0/1,0表示不是猫，1表示是猫)

    test_dataset=h5py.File('datasets/test_catvnoncat.h5','r') #保存的是促使集里面的图像数据(50张64*64的图像)
    test_set_x_orig=np.array(test_dataset["test_set_x"][:]) #保存的是测试集的图像对应的分类值(0/1,0表示不是猫，1表示是猫)
    test_set_y_orig=np.array(test_dataset["test_set_y"][:]) #保存的是以bytes类型保存的两个字符串数据，数据为:[b' non-cat' b' cat']

    classes=np.array(test_dataset["list_classes"][:])

    train_set_y_orig=train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig=test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

