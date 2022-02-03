import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten


def start_train(train_dataframe, test_dataframe, parameters):
    train_images, train_labels = adjust_training_dataset(train_dataframe, parameters)
    test_images, test_labels, test_prices = adjust_testing_dataset(test_dataframe, parameters)
    model = compile_model(parameters)
    fit_model(model, train_images, train_labels, parameters)
    prediction = predict_phase(model, test_images, test_labels, parameters)
    save_model(model)
    return prediction, test_labels, test_prices


def adjust_training_dataset(train_dataframe, parameters):
    train_images = train_dataframe.iloc[:, 2:].values
    train_labels = train_dataframe.iloc[:, 0]
    train_labels = keras.utils.to_categorical(train_labels, parameters["num_classes"])
    train_images = train_images.reshape(train_images.shape[0], parameters["input_w"], parameters["input_h"], 1)
    return train_images, train_labels


def adjust_testing_dataset(test_dataframe, parameters):
    test_images = test_dataframe.iloc[:, 2:].values
    test_labels = test_dataframe.iloc[:, 0]
    test_prices = test_dataframe.iloc[:, 1]
    test_labels = keras.utils.to_categorical(test_labels, parameters["num_classes"])
    test_images = test_images.reshape(test_images.shape[0], parameters["input_w"], parameters["input_h"], 1)
    return test_images, test_labels, test_prices


def compile_model(parameters):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(parameters["input_w"], parameters["input_h"], 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(parameters["num_classes"], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(), metrics=['accuracy', 'mae', 'mse'])

    return model


def fit_model(model, train_images, train_labels, parameters):
    model.fit(train_images, train_labels, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
              verbose=1, validation_data=None)


def predict_phase(model, test_images, test_label, parameters):
    prediction = model.predict(test_images, batch_size=parameters["batch_size"], verbose=1)
    print(model.evaluate(test_images, test_label, batch_size=parameters["batch_size"], verbose=1))
    return prediction


def save_model(model):
    model.save("my_model.h5")
