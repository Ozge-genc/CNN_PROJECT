from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Modeli yükle
model = load_model("model/flower_cnn_model.keras")

# Modelin tahmin edeceği sınıflar
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Yüklenen resimlerin kaydet
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        # 1. Formdan gelen dosyayı al ve kaydet
        file = request.files["file"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # 2. Resmi modelin istediği boyuta (128x128) getir
        img = image.load_img(image_path, target_size=(128, 128))

        # 3. Resmi sayısal verilere dök ve 0-1 arasına normalize et
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Tahmin yap ve en yüksek olasılıklı sınıfın adını al
        preds = model.predict(img_array)
        prediction = CLASS_NAMES[np.argmax(preds)]

    # Sonuçları HTML sayfasına gönder
    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)
