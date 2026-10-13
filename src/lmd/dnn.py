"""DNN models configured as in IJIS 2023, Table 12 and Sect. 5.2.1.

Ambiguities and the choices made here are listed in DEVIATIONS.
"""

import keras
from keras import layers, regularizers

DEVIATIONS = {
    "all": "Momentum is listed for Adam models (MLP 0.8, RNN 0.9); it has no Adam equivalent and is ignored.",
    "CNN": "Table 12 lists a Keras Embedding layer on continuous MinMax inputs, which Keras would truncate "
           "to integers; instead each feature is fed as one timestep of a (n_features, 1) sequence.",
    "LSTM": "Same Embedding substitution as CNN. 'Flatten: Yes' conflicts with GlobalAveragePooling1D in the "
            "text; GlobalAveragePooling1D is used.",
    "AE": "The paper does not specify how an autoencoder is used as a supervised classifier; implemented as a "
          "dense encoder-decoder stack (50/30/20/10/5 in the paper) with a softmax head.",
}

# name: (activation, optimizer, lr, dropout, loss, l2, hidden units, epochs, batch, standardize)
SPECS = {
    "MLP": ("relu", "adam", 1e-2, 0.15, "categorical_crossentropy", 1e-2, [30, 20, 10, 5], 40, 50, False),
    "CNN": ("relu", "sgd", 2e-2, 0.2, "categorical_crossentropy", 1e-2, [50, 30, 20, 15, 10, 5], 20, 32, True),
    "LSTM": ("tanh", "sgd", 2e-2, 0.3, "categorical_crossentropy", 2e-2, [50, 30, 20, 10, 5], 12, 30, True),
    "RNN": ("tanh", "adam", 1e-2, 0.15, "sparse_categorical_crossentropy", 1e-2, [30, 20, 10, 5], 40, 50, False),
    "AE": ("relu", "sgd", 1e-2, 0.2, "sparse_categorical_crossentropy", 1e-2, [50, 30, 20, 10, 5], 35, 50, False),
}


def _optimizer(kind: str, lr: float):
    if kind == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)


def build(name: str, n_features: int, n_classes: int = 3) -> keras.Model:
    act, opt, lr, drop, loss, l2, units, *_ = SPECS[name]
    reg = regularizers.l2(l2)
    inp = keras.Input(shape=(n_features,))
    x = inp
    if name == "MLP":
        for u in units:
            x = layers.Dense(u, activation=act, kernel_initializer="he_normal", kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(drop)(x)
    elif name == "AE":
        for u in units + units[-2::-1]:
            x = layers.Dense(u, activation=act, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(drop)(x)
    elif name == "CNN":
        x = layers.Reshape((n_features, 1))(x)
        for u in units:
            x = layers.Conv1D(u, 3, padding="same", activation=act, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(drop)(x)
    elif name in ("LSTM", "RNN"):
        cell = layers.LSTM if name == "LSTM" else layers.SimpleRNN
        x = layers.Reshape((n_features, 1))(x)
        for u in units:
            x = cell(u, activation=act, return_sequences=True, kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.Dropout(0.15)(x)  # Table 12 note: 0.15 dropout before the output layer
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name=name)
    model.compile(optimizer=_optimizer(opt, lr), loss=loss, metrics=["accuracy"])
    return model
