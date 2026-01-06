# CNN mimarisi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_model(num_classes):
    # Sıralı model yapısı
    model = Sequential([
        # 1. Katman: Basit özellikleri yakalar
        Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        MaxPooling2D(2,2),

        # 2. Katman: Karmaşık desenleri tanır
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),

        # 3. Katman: Veriyi düzleştirir
        Flatten(),

        # 4. Katman: Bilgileri birleştirip yorumla
        Dense(128, activation="relu"),

        # Son Katman: Kaç çiçek türü varsa o kadar çıkış üretir
        Dense(num_classes, activation="softmax")
    ])

    # Modeli derleme
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
