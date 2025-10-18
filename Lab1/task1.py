import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

st.title("Lab 1 - Dobrovolskyi Vladyslav")

grid_x = st.number_input("num rows (grid_x)", min_value=2, max_value=20, value=5)
grid_y = st.number_input("num collums (grid_y)", min_value=2, max_value=20, value=5)

uploaded_file = st.file_uploader("Upload BMP-image", type=["bmp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    w, h = img.size
    pixels = np.array(img)

    cell_w, cell_h = w // grid_x, h // grid_y
    features = []
    for i in range(grid_y):
        for j in range(grid_x):
            region = pixels[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            black_pixels = np.sum(region < 128)
            features.append(int(black_pixels))

    s = sum(features)
    norm_sum = [round(f/s, 3) if s > 0 else 0 for f in features]

    m = max(features)
    norm_mod = [round(f/m, 3) if m > 0 else 0 for f in features]

    st.subheader("Absolute vector")
    st.write(" ".join(str(f) for f in features))

    st.subheader("Normalized (summ)")
    st.write(" ".join(str(f) for f in norm_sum))

    st.subheader("Normalized (module)")
    st.write(" ".join(str(f) for f in norm_mod))

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")

    for i in range(1, grid_x):
        ax.axvline(i*cell_w, color="red", linewidth=1)
    for j in range(1, grid_y):
        ax.axhline(j*cell_h, color="red", linewidth=1)

    st.pyplot(fig)
