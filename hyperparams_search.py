from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np

# Dataset settings
batch_size = 32
img_width = 128
img_height = 128
img_channels = 3
train_dir = './train'
test_dir = './test'

# Load datasets
train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='both',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    seed=None,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)

# Define the model builder function
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, img_channels)),
        
        # First Conv layer: Tune number of filters and kernel size
        keras.layers.Conv2D(
            filters=hp.Int('filters_1', min_value=16, max_value=64, step=16),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(2, 2),
        
        # Second Conv layer
        keras.layers.Conv2D(
            filters=hp.Int('filters_2', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(2, 2),

        # Third Conv layer
        keras.layers.Conv2D(
            filters=hp.Int('filters_3', min_value=64, max_value=256, step=64),
            kernel_size=hp.Choice('kernel_size_3', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Flatten(),
        
        # Fully connected layer: Tune the number of neurons
        keras.layers.Dense(
            units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
            activation='relu'
        ),
        
        # Tune dropout rate
        keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)),
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Tune the learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Initialize Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='tuner_results',
    project_name='image_classification'
)

# Callback to stop early if validation loss does not improve
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Perform hyperparameter tuning
tuner.search(train_ds, validation_data=val_ds, epochs=10, callbacks=[stop_early])

# Get the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print best hyperparameters
print(f"Best Hyperparameters:\n"
      f"Filters (1st Conv): {best_hps.get('filters_1')}\n"
      f"Filters (2nd Conv): {best_hps.get('filters_2')}\n"
      f"Filters (3rd Conv): {best_hps.get('filters_3')}\n"
      f"Kernel Size (1st Conv): {best_hps.get('kernel_size_1')}\n"
      f"Kernel Size (2nd Conv): {best_hps.get('kernel_size_2')}\n"
      f"Kernel Size (3rd Conv): {best_hps.get('kernel_size_3')}\n"
      f"Dense Units: {best_hps.get('dense_units')}\n"
      f"Dropout Rate: {best_hps.get('dropout')}\n"
      f"Learning Rate: {best_hps.get('learning_rate')}")

# Train the best model
best_model = tuner.hypermodel.build(best_hps)

history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[stop_early]
)

# Evaluate on test set
test_loss, test_acc = best_model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the best model
best_model.save("best_pneumonia_model.keras")

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()