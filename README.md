# 🎨 AI Seasonal Color & Undertone Analyzer

An end-to-end computer vision prototype built for the AI Internship Assignment (BabyDino / Zenith / J.e.r.k).

## 🚀 Live Demo
**Try the working app here:** [Your Streamlit App Link]

---

## 🛠️ Architecture & Pipeline
1. **Face & Region Detection:** Uses OpenCV Haar Cascades to locate facial boundaries and crops non-occluded skin regions (forehead/cheeks).
2. **Undertone Extraction:** Applies K-Means clustering ($K=3$) on pixel color arrays to isolate dominant skin RGB values.
3. **Seasonal Classification:** Converts RGB $\rightarrow$ HSV color space to evaluate undertone warmth/coolness and value depth, matching users to seasonal color palettes.

---

## 📦 Local Installation
```bash
git clone [https://github.com/](https://github.com/(https://github.com/akshatpatidar23/ai-color-analysis).git
cd <ai-color-analysi>
pip install -r requirements.txt
streamlit run app.py
