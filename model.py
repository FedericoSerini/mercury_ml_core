import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
import matplotlib.pyplot as plt


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
                  optimizer=keras.optimizers.Nadam(), metrics=['accuracy', 'mae', 'mse'])

    # RMSprop [0.4856254458427429, 0.8214285969734192, 0.12258549779653549, 0.08360042423009872]
    # RMSprop [0.2817503809928894, 0.875, 0.08933816850185394, 0.05786341428756714]
    # RMSprop [0.6731988191604614, 0.8243727684020996, 0.1198042705655098, 0.08776666224002838]

    # Adam [0.5218594074249268, 0.8428571224212646, 0.11165329068899155, 0.08420402556657791]
    # Adam [0.8530320525169373, 0.8428571224212646, 0.10867674648761749, 0.08490855246782303]
    # Adam [0.6285068988800049, 0.8428571224212646, 0.11547085642814636, 0.0857032835483551]

    # Adamax [0.5503571629524231, 0.7928571701049805, 0.1686561107635498, 0.10972392559051514]
    # Adamax [0.7007958889007568, 0.7992831468582153, 0.15214642882347107, 0.10711833089590073]
    # Adamax [0.5606870651245117, 0.7535714507102966, 0.176607146859169, 0.11240770667791367]

    # Nadam [0.5084196329116821, 0.8678571581840515, 0.09934310615062714, 0.06840437650680542]
    # Nadam [0.526914656162262, 0.8464285731315613, 0.10316471010446548, 0.07353658974170685]
    # Nadam [0.5987406373023987, 0.8637992739677429, 0.10102051496505737, 0.07199636101722717]

    return model


def fit_model(model, train_images, train_labels, parameters):
    history = model.fit(train_images, train_labels, batch_size=parameters["batch_size"], epochs=parameters["epochs"],
              verbose=1, validation_data=None)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predict_phase(model, test_images, test_label, parameters):
    prediction = model.predict(test_images, batch_size=parameters["batch_size"], verbose=1)
    print(model.evaluate(test_images, test_label, batch_size=parameters["batch_size"], verbose=1))
    return prediction


def save_model(model):
    model.save("my_model.h5")
