# Eğitim
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.model import create_model

def train_model():
    # 1. Görüntü hazırlayıcıyı tanımla
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # 2. Eğitim ve test veri setlerini oluştur
    train_data = datagen.flow_from_directory(
        "data/raw/flowers",
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )

    # 3. Test verilerini yükle
    val_data = datagen.flow_from_directory(
        "data/raw/flowers",
        target_size=(128,128),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    # 4. Modeli oluştur
    model = create_model(train_data.num_classes)

    # 5. Eğitimi başlat
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    # 6. Modeli kaydet
    model.save("../model/flower_cnn_model.keras")

