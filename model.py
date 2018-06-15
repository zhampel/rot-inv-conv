from __future__ import print_function

try:
    import keras
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from layers import Convolution2D_4 as conv2d_4rot
except ImportError as e:
    print(e)
    raise ImportError

conv_dict = {'conv2d'      : Conv2D,
             'conv2d_4rot' : conv2d_4rot}


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def model(dir_struct=None, train_gen=None, valid_gen=None, epochs=-1, layer_string_list=[]):

    # Provide some basic numbers to the model
    batch_size  = train_gen.batch_size
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes

    # Default is to train over whole data set
    if (epochs == -1):
        epochs = int(train_gen.n / batch_size)

    # Callbacks
    history = History()
    csv_log = keras.callbacks.CSVLogger(dir_struct.log_file)

    ## Model
    # Get requested layer order
    if len(layer_string_list) > 3:
        raise ValueError('Requested {} layers. Currently only ' \
                         'supporting 3 convolutional layers.'.format(len(layer_string_list)))

    conv_layers = [ conv_dict[layer_string] for layer_string in layer_string_list ]

    # Build model layers
    model = Sequential()
    model.add(conv_layers[0](32, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(conv_layers[1](64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(conv_layers[2](64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model 
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    # Fit model using generator
    model.fit_generator(train_gen,
                        steps_per_epoch=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=valid_gen,
                        validation_steps=10,
                        class_weight=None,
                        max_q_size=1000,
                        callbacks=[history, csv_log])
    
    # Save model to JSON
    print("\nSaving model to directory {}".format(dir_struct.main_dir))
    model_json = model.to_json()
    with open(dir_struct.model_file, 'w') as json_file:
        json_file.write(model_json)
    print("Saved model to disk:\t{}".format(dir_struct.model_file))

    # Save weights to HDF5
    model.save_weights(dir_struct.weights_file)
    print("Saved weights to disk:\t{}\n".format(dir_struct.weights_file))
    
    return history, model
