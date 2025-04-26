import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

def build_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Construit un modèle CNN personnalisé.
    """
    model = Sequential()

    # Bloc convolutif 1
    model.add(Conv2D(64, (5, 5), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc convolutif 2
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Bloc convolutif 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Couches fully connected
    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compiler le modèle
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(X_train, y_train, X_test, y_test, model_path='../models/emotion_model.h5'):
    """
    Entraîne le modèle et le sauvegarde.
    """
    model = build_model()

    # Callback pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True)

    # Entraîner le modèle
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=64,
        epochs=20,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint]
    )
    print("Modèle entraîné et sauvegardé.")

if __name__ == "__main__":
    # Charger les données prétraitées
    X_train = np.load('../Data/X_train.npy')
    y_train = np.load('../Data/y_train.npy')
    X_test = np.load('../Data/X_test.npy')
    y_test = np.load('../Data/y_test.npy')

    # Entraîner le modèle
    train_model(X_train, y_train, X_test, y_test)