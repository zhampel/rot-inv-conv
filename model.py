from __future__ import print_function

try:
    import keras
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Sequential
except ImportError as e:
    print(e)
    raise ImportError

epochs = 10

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
    
def model(dir_struct=None, train_gen=None, valid_gen=None):

    # Provide some basic numbers to the model
    batch_size  = train_gen.batch_size
    input_shape = train_gen.image_shape
    num_classes = train_gen.num_classes

    # Callbacks
    history = History()
    csv_log = keras.callbacks.CSVLogger(dir_struct.log_file)

    # Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
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
                        callbacks=[history, csv_log])
    
    # Save model to JSON
    model_json = model.to_json()
    with open(dir_struct.model_file, 'w') as json_file:
        json_file.write(model_json)
    print("Saved model to disk: {}".format(dir_struct.model_file))

    # Save weights to HDF5
    model.save_weights(dir_struct.weights_file)
    print("Saved weights to disk: {}".format(dir_struct.weights_file))
    
    return history, model
