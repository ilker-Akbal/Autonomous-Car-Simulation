## 📊 Dataset

Bu projede **BDD100K (Berkeley DeepDrive 100K)** veri seti kullanılmıştır.  
BDD100K, otonom sürüş araştırmaları için geliştirilmiş geniş ölçekli bir veri setidir.

### 📌 Kullanım ve Lisans Notu

- **BDD100K veri seti bu repoya dahil değildir.**
- Bu repository içerisinde yer alan görseller, veri setine ait ham görüntüler değil; model tarafından üretilmiş **inference ve debug çıktılarıdır.**
- Veri setine erişim, yalnızca resmi kaynak üzerinden ve ilgili lisans/ kullanım şartları kabul edilerek sağlanmalıdır.

### 🔗 Resmi Kaynak

https://bdd-data.berkeley.edu/

### ⚠️ Önemli

Bu projeyi kullanan kullanıcılar, BDD100K veri setini indirirken ve kullanırken ilgili **lisans ve kullanım koşullarına uymakla sorumludur.**

---

## 🛠️ Kurulum

```bash
git clone https://github.com/username/autonomous_driving.git
cd autonomous_driving
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### Görüntü inference

```bash
python scripts/infer_image.py
```

### Video inference

```bash
python scripts/video_processor.py
```

---

## 📁 Proje Yapısı

```
autonomous_driving/
│
├── src/            # ana modüler kod
├── scripts/        # çalıştırma scriptleri
├── configs/        # config dosyaları
├── notebooks/      # deney ve analiz
├── outputs/        # model ve sonuçlar
└── main.py
```

---

## 🧪 Eğitim

```bash
python scripts/train_yolo.py
```

Model çıktıları:

```
outputs/models/
```

---

## 🛣️ Lane Analizi

- Şerit çizgileri tespit edilir
- Araç pozisyonu analiz edilir
- Basit steering bilgisi üretilebilir

---

## 🖼️ Çıktılar

- Bounding box + label
- Trafik ışığı durumu
- Ön araç mesafesi
- Risk seviyesi
- Karar çıktısı
- Debug overlay

---

## 🔮 Gelecek Çalışmalar

- Gazebo entegrasyonu
- CARLA simülasyonu
- RL tabanlı karar sistemi
- Multi-camera destek

---

## 📌 Özet

Bu proje, görüntü işleme ile çevreyi algılayıp üzerine karar mekanizması ekleyerek  
**otonom sürüş davranışını simüle eden bir ADAS sistemidir.**
