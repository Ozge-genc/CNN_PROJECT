# Eğitim
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import create_model

def train_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        "data/raw/flowers",
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    val_data = datagen.flow_from_directory(
        "data/raw/flowers",
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    model = create_model(train_data.num_classes)

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    model.save("flower_cnn_model.keras")

