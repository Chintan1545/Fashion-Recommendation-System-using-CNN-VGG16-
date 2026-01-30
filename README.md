# 👗 Fashion Recommendation System using CNN (VGG16)

A **content-based fashion recommendation system** that suggests visually similar clothing items using **deep learning (CNN)** and **cosine similarity**.  
The system extracts deep features from images using a **pre-trained VGG16 model** and recommends similar fashion items based on visual appearance.

---

## 🚀 Features
- Uses **VGG16 (ImageNet)** as a feature extractor
- Computes similarity using **cosine similarity**
- Works on **CPU (GPU optional)**
- Built using **TensorFlow + Conda**
- Easy to extend with FAISS or Streamlit

---

## 🧠 Tech Stack
- Python 3.9
- TensorFlow / Keras
- VGG16 (CNN)
- NumPy, SciPy
- OpenCV, Pillow
- Matplotlib

---

## 📂 Project Structure
```bash
fashion-recommendation/
│
├── dataset/
│ └── women_fashion/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
```


---

## ⚙️ Environment Setup (Conda)

```bash
conda create -n fashion_reco python=3.9 -y
conda activate fashion_reco
pip install -r requirements.txt
```

---

## 📦 Requirements
```bash
tensorflow
numpy
scipy
opencv-python
pillow
matplotlib
scikit-learn
```

---

## ▶️ How It Works

1. Load fashion images from a directory.
2. Resize images to 224×224
3. Extract deep features using VGG16 (without top layers)
4. Normalize feature vectors
5. Compute cosine similarity
6. Recommend top-N visually similar images

---

## ▶️ Run the Project
1️⃣ Set Image Directory 
```bash
IMAGE_DIR = r"D:\datasets\women_fashion\women fashion"
```
2️⃣ Run Feature Extraction
```bash
python fashion_recommendation.py
```
3️⃣ Get Recommendations
```bashrecommend_fashion_items(
    input_image_path="path/to/input_image.jpg",
    image_paths=image_paths,
    features=all_features,
    top_n=4
)
```

---

## 📸 Sample Output

The system displays:
- Input fashion image
- Top-N visually similar fashion recommendations

---

## 🧪 Model Details

- Model: VGG16
- Weights: ImageNet
- Input Size: 224 × 224 × 3
- Similarity Metric: Cosine Similarity

---

## 💡 Use Cases

- Fashion e-commerce recommendations
- Visual product search
- Style similarity detection
- AI-based shopping assistants

---

## 🔮 Future Enhancements

- Integrate FAISS for fast similarity search
- Add Streamlit web UI
- Category-aware recommendations
- Deploy using Docker

--- 

## 👨‍💻 Author

Chintan Dabhi
MCA Student | AI & ML Enthusiast
