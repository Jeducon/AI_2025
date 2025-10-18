import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Lab 2 - identification of image")


def feature_vector(img, grid_x, grid_y, thr=128):
    img = img.convert("L")
    w, h = img.size
    pixels = np.array(img)
    cell_w, cell_h = w // grid_x, h // grid_y

    features = []
    for i in range(grid_y):
        for j in range(grid_x):
            region = pixels[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            black_pixels = np.sum(region < thr)
            features.append(int(black_pixels))

    s = sum(features)
    m = max(features)
    norm_sum = [f/s if s > 0 else 0 for f in features]
    norm_mod = [f/m if m > 0 else 0 for f in features]
    return features, norm_sum, norm_mod


def distance(x, y, metric="Euclidic"):
    if metric == "Euclidic":
        return math.sqrt(sum((a-b)**2 for a, b in zip(x, y)))
    elif metric == "Manhattan":
        return sum(abs(a-b) for a, b in zip(x, y))
    elif metric == "Chebishev":
        return max(abs(a-b) for a, b in zip(x, y))
    else:
        return None


def show_grid(img, grid_x, grid_y):
    w, h = img.size
    cell_w, cell_h = w // grid_x, h // grid_y
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for i in range(1, grid_x):
        ax.axvline(i * cell_w, color="red", linewidth=1)
    for j in range(1, grid_y):
        ax.axhline(j * cell_h, color="red", linewidth=1)
    st.pyplot(fig)


grid_x = st.number_input("num of collumn (grid_x)", 2, 20, 5)
grid_y = st.number_input("num of rows (grid_y)", 2, 20, 5)
thr = st.slider("Edge of black pixel", 0, 255, 128)
metric = st.radio("Choose normalization:", ["Euclidic", "Manhattan", "Chebishev"])

st.divider()


st.subheader("Enter standart BMP")

etalons = {}
uploaded_refs = st.file_uploader("Download standart BMP", type=["bmp"], accept_multiple_files=True)

if uploaded_refs:
    for f in uploaded_refs:
        img = Image.open(f)
        st.image(img, caption=f.name, width=150)
        show_grid(img, grid_x, grid_y)
        _, _, norm_mod = feature_vector(img, grid_x, grid_y, thr)
        etalons[f.name] = norm_mod
    st.success("Standart images downloaded!")

st.divider()


st.subheader("Clasification")

unknown_file = st.file_uploader("Download new BMP", type=["bmp"])

if unknown_file and etalons:
    unknown_img = Image.open(unknown_file)
    st.image(unknown_img, caption="Unknown image", width=200)
    show_grid(unknown_img, grid_x, grid_y)
    _, _, unknown_vec = feature_vector(unknown_img, grid_x, grid_y, thr)

    st.write("**Normalized vector:**")
    st.write(" ".join(str(round(v, 3)) for v in unknown_vec))

    st.write("### Result of comparison:")
    results = {}
    for name, ref_vec in etalons.items():
        d = distance(unknown_vec, ref_vec, metric)
        results[name] = d
        st.write(f"{name}: dist = {round(d, 4)}")

    best_match = min(results, key=results.get)
    st.success(f"Unknown image belongs to: **{best_match}**")

elif unknown_file and not etalons:
    st.warning("Download standart images first!")

