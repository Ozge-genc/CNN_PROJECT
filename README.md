# 🌸 CNN ile Çiçek Sınıflandırma Projesi

## 📌 Proje Açıklaması

Bu projede, **Yapay Sinir Ağları (Convolutional Neural Networks - CNN)** kullanılarak çiçek görsellerinin otomatik olarak sınıflandırılması amaçlanmıştır. Model bir çiçeğin görseline bakarak onu aşağıdaki beş sınıftan birine atamaktadır:

* Daisy (Papatya)
* Dandelion (Karahindiba)
* Rose (Gül)
* Sunflower (Ayçiçeği)
* Tulip (Lale)

Proje, derin öğrenme ve görüntü işleme alanlarında CNN mimarisinin temel çalışma mantığını göstermek amacıyla geliştirilmiştir.

---

## 📂 Veri Seti

Veri seti **Kaggle** üzerinden temin edilmiştir:

**Flowers Dataset – imsparsh**

* Toplam görsel sayısı: ~4300
* Sınıf sayısı: 5
* Görseller renkli (RGB) ve farklı çözünürlüklerdedir

Veri seti eğitim ve doğrulama (validation) olarak ayrılarak kullanılmıştır.

---

## 🛠️ Kullanılan Teknolojiler

* **Python 3.10**
* **TensorFlow & Keras**
* **Convolutional Neural Networks (CNN)**
* **Conda Sanal Ortamı**

---

## 🧠 Model Mimarisi

Model, temel bir CNN mimarisi üzerine kurulmuştur:

* **Conv2D katmanları** + **ReLU** 
* **MaxPooling**
* **Fully Connected (Dense)**
* **Softmax**

---

## 📊 Eğitim Sonuçları

* Eğitim (Training) Doğruluğu: **~%99**
* Doğrulama (Validation) Doğruluğu: **~%63**

Eğitim ve doğrulama doğrulukları arasındaki fark, modelde **overfitting (aşırı öğrenme)** problemi olduğunu göstermektedir.

> Bu durum; veri artırma (data augmentation), dropout, erken durdurma (early stopping) veya daha sade bir model mimarisi ile azaltılabilir.

---

## ▶️ Projenin Çalıştırılması

Aşağıdaki adımları takip ederek projeyi kendi bilgisayarınızda çalıştırabilirsiniz:

```bash
conda create -n proje python=3.10
conda activate proje
pip install -r requirements.txt
python main.py
```