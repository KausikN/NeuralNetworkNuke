# Imports
from Attacks.FGSM import FGSM
from Models import BasicFNNModel
from Utils import OneHotEncode
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.datasets import mnist
from torch import optim
import random
import numpy as np
import matplotlib.pyplot as plt
# Imports

# Set Parameters
img_rows, img_cols = 28, 28
Retrain = True
model_path = input("Enter prerrained model path (Enter None to train new model): ")
# Set Parameters

# Load Data
print("Started Loading Data from mnist Dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Finished Loading Data from mnist Dataset")

print("Started Preprocessing")
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = np.array(x_train).reshape(60000, img_rows*img_cols)

print("X_train", x_train.shape)
print("Y_train", y_train.shape)
print("X_test", x_test.shape)
print("Y_test", y_test.shape)

x_train = torch.from_numpy(x_train)
x_test = torch.from_numpy(x_test)
y_train = torch.from_numpy(y_train.astype(int))
y_test = torch.from_numpy(y_test.astype(int))

print("Finished Preprocessing")
# Load Data

# Model
model = None
if (Retrain or not os.path.exists(model_path)):
    act_funcs = [nn.ReLU()]
    out_act_funcs = [nn.Softmax(dim=1)]
    model = BasicFNNModel(img_rows*img_cols, 10, [64, 128, 256, 128, 64], act_funcs, out_act_funcs)
    # Training
    epochs = 10000
    lr = 0.01
    momentum = 0.9
    loss_fn = F.cross_entropy
    opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_arr = model.fit(x_train, OneHotEncode(y_train), opt, loss_fn, epochs, True)

    print("Losses: ", loss_arr)
    print("MinLoss: ", model.min_loss, " - ", model.min_loss_config.min_loss)
    print("Final Epoch: ", model.cur_iter)
    print("Best Epoch: ", model.min_loss_config.cur_iter)

    model_path = model_path
    if (os.path.exists(model_path)):
        os.remove(model_path)
    pickle.dump(model.min_loss_config, open(model_path, "wb"))
    print("Model Trained and Saved")

model = pickle.load(open(model_path, "rb")) # Load Best Model
print("Loaded Best Model")


print("Choose what to do: ")
print("0    -   Attack")
print("1    -   Defence")

function_choice = input("Enter choice: ")

if function_choice == '0':
    print("Choose Method of FGSM: ")
    print("0    -   Normal/Targeted")
    print("1    -   Iterative")

    type_choice = input("Enter choice: ")

    if type_choice == '0':
        stop_display = False
        while (not stop_display):
            n_images = 2

            for i in range(1, n_images+1):
                randindex = random.randint(0, np.array(x_test).shape[0]-1)
                x = np.array(x_test[randindex])
                x_flattened = torch.from_numpy(x.reshape(1, img_rows*img_cols))
                y_pred = torch.from_numpy(np.array(y_test[randindex]).reshape(-1))

                att_fgsm = FGSM()
                # x_adv_flattened - new image, h_adv - predicted for x_adv_flattened, h - predicted for original x
                x_adv_flattened, h_adv, h = att_fgsm.fgsm(x_flattened, y_pred, model.net, model.loss_fn, False)

                x_flattened = np.array(x_flattened)
                x_adv = x_adv_flattened.reshape(img_rows, img_cols)

                added_noise_flattened = x_adv_flattened - x_flattened
                added_noise = added_noise_flattened.reshape(img_rows, img_cols)

                print("Shapes: x: ", x.shape, ", x_flattened: ", x_flattened.shape, 
                ", x_adv: ", x_adv.shape, ", x_adv_flattened: ", x_adv_flattened.shape)

                print("Y_pred before attack: ", str(np.argmax(h)))
                print("Y_pred after attack: ", str(np.argmax(h_adv)))
                print("Y_pred before attack: ", str(h))
                print("Y_pred after attack: ", str(h_adv))

                columns = 3
                rows = 1
                fig = plt.figure(figsize=(8, 8))
                fig.add_subplot(rows, columns, 1)
                plt.imshow(x[0])
                fig.add_subplot(rows, columns, 1+1)
                plt.imshow(added_noise)
                fig.add_subplot(rows, columns, 1+2)
                plt.imshow(x_adv)
                plt.show()

            stop_display = input("Continue? ") not in ['', 'Y', 'yes', 'y']

