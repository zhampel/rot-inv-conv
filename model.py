from __future__ import print_function

try:
    import keras
    from keras.layers import Dense, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.models import Sequential
    from load_data import load_img, img_generator
except ImportError as e:
    print(e)
    raise ImportError

# Specifics
batch_size = 128
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
    
def model(input_shape, num_classes, train_gen, valid_gen):
    ## Data
    #img, label = load_img(filepath, img_id=10, file_type='png')
    #input_shape = (img.shape[0], img.shape[1], 1)
    #num_classes=10

    # History
    history = History()

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

    # Fit using generator
    model.fit_generator(train_gen,
                        steps_per_epoch=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=valid_gen,
                        validation_steps=10,
                        class_weight=None,
                        callbacks=[history])
    
    return history, model
