import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import urllib.request
import os

st.set_page_config(page_title="AI Color & Undertone Analyzer", layout="centered")

st.title("🎨 AI Seasonal Color & Undertone Analyzer")
st.write("Upload a portrait/selfie to extract skin undertones using K-Means clustering and detect your seasonal color palette.")

# Download face detection model
cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

uploaded_img = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) == 0:
        h, w = img_rgb.shape[:2]
        skin_patch = img_rgb[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
    else:
        x, y, w, h = faces[0]
        skin_patch = img_rgb[y+int(h*0.5):y+int(h*0.7), x+int(w*0.35):x+int(w*0.65)]
        
    pixels = skin_patch.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dom_color = np.uint8(kmeans.cluster_centers_[np.argmax(counts)])
    
    pixel = np.uint8([[dom_color]])
    hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv_pixel
    
    is_warm = (h > 8 and h < 25)
    is_light = (v > 140)
    
    if is_warm and is_light:
        season, palette = "Spring (Warm & Light)", ['#F9E076', '#FF9C7D', '#75C1A3', '#9D8461']
    elif not is_warm and is_light:
        season, palette = "Summer (Cool & Light)", ['#A3C1AD', '#C9A0DC', '#F4C2C2', '#778899']
    elif is_warm and not is_light:
        season, palette = "Autumn (Warm & Dark)", ['#C04000', '#DAA520', '#556B2F', '#8B4513']
    else:
        season, palette = "Winter (Cool & Dark)", ['#000080', '#E30B5C', '#FFFFFF', '#000000']

    st.success(f"**Detected Season:** {season}")
    skin_hex = f"#{dom_color[0]:02x}{dom_color[1]:02x}{dom_color[2]:02x}".upper()
    
    cols = st.columns(5)
    with cols[0]:
        st.write("**Skin Tone**")
        st.markdown(f"<div style='background-color:{skin_hex};height:50px;border-radius:6px;border:1px solid #ccc;'></div>", unsafe_allow_html=True)
        st.caption(skin_hex)
        
    for i, hex_c in enumerate(palette):
        with cols[i+1]:
            st.write(f"**Palette {i+1}**")
            st.markdown(f"<div style='background-color:{hex_c};height:50px;border-radius:6px;border:1px solid #ccc;'></div>", unsafe_allow_html=True)
            st.caption(hex_c)
