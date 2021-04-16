from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import time
from tqdm import trange
import torch
import shelve

train_acc = tf.metrics.CategoricalAccuracy()
gamma_set = [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0]

def build_model(width, height, depth, classes):
    # initialize the input shape and channels dimension to be
    # "channels last" ordering
    inputShape = (height, width, depth)
    # build the model using Keras' Sequential API
    model = Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=inputShape,
               kernel_initializer=GlorotNormal()),
        Conv2D(32, (3, 3), padding="valid", activation="relu", kernel_initializer=GlorotNormal()),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=100, activation="relu", kernel_initializer=GlorotNormal()),
        Dense(units=200, kernel_initializer=GlorotNormal()),
        Dense(units=classes, activation="softmax", kernel_initializer=GlorotNormal())

    ])
    # return the built model to the calling function
    return model


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    t = trange(0, len(inputs) - batchsize + 1, batchsize)
    for start_idx in t:
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], t


def step(X, y):
    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the
        # loss
        pred = model(X, training=True)
        # Compute the loss and the initial gradient
        loss = categorical_crossentropy(y, pred)
        train_acc.update_state(y, pred)
    # calculate the gradients using our tape and then update the
    # model weights
    grads = tape.gradient(loss, model.trainable_variables)
    # opt.apply_gradients(zip(grads, model.trainable_variables))
    return grads


def choose_subset_update(grads_list,layer_gamma_list):
    new_grads_list = []
    layer_index = 0
    totalnum = 0
    for idx in range(len(grads_list[0])):
        new_grads = torch.zeros(grads_list[0][idx].numpy().shape)
        average_matrix = torch.zeros(grads_list[0][idx].numpy().shape)
        if layer_index % 2 == 0:
            layer_gamma = layer_gamma_list[layer_index//2]
        elem_num = int(grads_list[0][idx].shape[-1] * layer_gamma)
        for node_idx in range(node_num):
            grads_copy = torch.tensor(grads_list[node_idx][idx].numpy())
            grads_subset, index = torch.topk(torch.abs(grads_copy), elem_num, dim=-1,largest=True)
            grads_copy -= grads_copy.scatter(dim=-1, index=index, src=torch.zeros_like(grads_subset))
            average_matrix.scatter_add_(dim=-1, index=index, src=torch.ones_like(grads_subset))
            new_grads += grads_copy
        totalnum += grads_subset.numel()
        layer_index += 1

        average_matrix = torch.where(average_matrix == 0, torch.tensor(1, dtype=average_matrix.dtype),average_matrix)
        new_grads /= average_matrix
        new_grads_list.append(tf.convert_to_tensor(new_grads.numpy()))

    opt.apply_gradients(zip(new_grads_list, model.trainable_variables))
    return totalnum

def choose_gamma():
    variance = []
    #new_gamma = list(range(len(gamma)))
    new_gamma = []
    for layer_index in range(0,len(model.trainable_variables),2):
        variance.append(np.var(model.trainable_variables[layer_index].numpy(), ddof=1))
    for i in range(len(variance)):
        new_gamma.append(np.around(variance[i]/np.sum(variance),2))
    variance_index = np.argsort(np.array(variance))[::-1]
    #for l in range(len(variance)):
        #new_gamma[variance_index[l]] = gamma[l]
    return new_gamma
def adjust_gamma(epoch):
    new_gamma_list = gamma_set[epoch//10:5+(epoch//10)]
    return new_gamma_list


batch_size = 64
num_classes = 10
epoch_num = 40
learning_rate = 0.001
node_num = 4
# -----------------------------------------------------------------------------
"Data Loading"
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# -----------------------------------------------------------------------------
"Data Preprocessing"
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# -----------------------------------------------------------------------------
"Initializing our model"
model = build_model(32, 32, 3, 10)
opt = SGD(learning_rate=learning_rate,decay=learning_rate/epoch_num)
# model.compile(optimizer=opt, loss=categorical_crossentropy, metrics=["acc"])

# compute the number of batch updates per epoch
numUpdates = int(x_train.shape[0] / batch_size)
# loop over the number of epochs
loss = []
acc = []
data_quantity = []
for epoch in range(epoch_num):
    # show the current epoch number
    # loop over the data in batch size increments
    #if(epoch % 10 == 0) and epoch <= (len(gamma_set) - 5 + 1) * 10:
    #gamma = adjust_gamma(epoch)
    a0 = []
    l0 = []
    epochStart = time.time()
    quantity = 0
    for x_batch, y_batch, t in iterate_minibatches(x_train, y_train, batchsize=batch_size, shuffle=True):
        grads_list = []
        x_batch_split_list = np.split(x_batch, node_num)
        y_batch_split_list = np.split(y_batch, node_num)
        layer_gamma_list = choose_gamma()
        for idx in range(node_num):
            grads = step(x_batch_split_list[idx], y_batch_split_list[idx])
            grads_list.append(grads)
            a1 = train_acc.result().numpy()
            a0.append(a1)

        quantity += choose_subset_update(grads_list,layer_gamma_list)
        t.set_description("Epoch: " + str(epoch + 1) + '/' + str(epoch_num) + " Acc:" + str(f'{a1: 0.3f}'))

    acc.append(np.mean(a0))
    data_quantity.append(quantity)

epochEnd = time.time()
lapsed = (epochEnd - epochStar  t) / 60.0
print("took {:.4} minutes".format(elapsed))

overall_acc = np.asarray(acc).flatten()
my_shelf = shelve.open('E:\\data\\data' + 'shelve_accuracy_GradientTape1' + '_performance' + '.out')
my_shelf['data'] = {'acc': overall_acc,'data_quantity': data_quantity}
my_shelf.close()
