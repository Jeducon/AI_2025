import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Lab 3")

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
    m = max(features) if features else 0
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

grid_x = st.number_input("num of columns (grid_x)", 2, 20, 5)
grid_y = st.number_input("num of rows (grid_y)", 2, 20, 5)
thr = st.slider("edge of black pixel", 0, 255, 128)
metric = st.radio("choose metric:", ["Euclidic", "Manhattan", "Chebishev"])
num_classes = st.number_input("number of classes", 3, 12, 5)

st.divider()
st.subheader("Enter training BMPs per class")

class_prototypes = {}
tabs = st.tabs([f"class {i+1}" for i in range(num_classes)])
for idx, t in enumerate(tabs):
    with t:
        cname = st.text_input("class name", value=f"class_{idx+1}", key=f"name_{idx}")
        uploaded = st.file_uploader("download class BMPs (~10)", type=["bmp"], accept_multiple_files=True, key=f"files_{idx}")
        vectors = []
        if uploaded:
            for f in uploaded:
                img = Image.open(f)
                st.image(img, caption=f"{cname}: {f.name}", width=150)
                show_grid(img, grid_x, grid_y)
                _, _, v = feature_vector(img, grid_x, grid_y, thr)
                vectors.append(v)
        if vectors:
            proto = np.mean(np.array(vectors), axis=0).tolist()
            class_prototypes[cname] = proto
            st.write("prototype:", " ".join(str(round(x,3)) for x in proto))
        else:
            st.write("prototype: —")

st.divider()
st.subheader("Classification")

unknown_file = st.file_uploader("download unknown BMP", type=["bmp"])

if unknown_file and class_prototypes:
    unknown_img = Image.open(unknown_file)
    st.image(unknown_img, caption="unknown image", width=200)
    show_grid(unknown_img, grid_x, grid_y)
    _, _, unknown_vec = feature_vector(unknown_img, grid_x, grid_y, thr)
    st.write("normalized vector:", " ".join(str(round(v, 3)) for v in unknown_vec))

    st.write("result of comparison:")
    results = {}
    for name, proto_vec in class_prototypes.items():
        d = distance(unknown_vec, proto_vec, metric)
        results[name] = d
        st.write(f"{name}: dist = {round(d, 6)}")
    best_match = min(results, key=results.get)
    st.success(f"unknown image belongs to: {best_match}")

elif unknown_file and not class_prototypes:
    st.warning("download training images first")


