from tensorflow.keras.models import Model
from tensorflow.keras.layers\
    import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from dataset import IMAGE_SIZE


def convnet():
    inputs = Input(shape=(*IMAGE_SIZE, 1))

    hiddens = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', activation='relu')(inputs)
    hiddens = MaxPool2D(pool_size=2)(hiddens)

    hiddens = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', activation='relu')(hiddens)
    hiddens = MaxPool2D(pool_size=2)(hiddens)

    hiddens = Conv2D(32, kernel_size=3, strides=1,
                     padding='same', activation='relu')(hiddens)
    hiddens = MaxPool2D(pool_size=2)(hiddens)

    fullcs = Flatten()(hiddens)
    fullcs = Dense(units=512, activation='relu')(fullcs)

    outputs = Dense(units=3, activation='softmax')(fullcs)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['acc'])

    return model
