import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import math

st.title("Lab 4 Perceptron")

def edges(w, h, gx, gy):
    xs = [round(i*w/gx) for i in range(gx+1)]
    ys = [round(j*h/gy) for j in range(gy+1)]
    return xs, ys

def feat_abs(img, gx, gy, thr):
    a = np.array(img.convert("L"))
    h, w = a.shape
    xs, ys = edges(w, h, gx, gy)
    f = []
    for i in range(gy):
        for j in range(gx):
            r = a[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            f.append(int(np.sum(r < thr)))
    return f

def feat_norm(f, mode):
    s = sum(f)
    m = max(f) if f else 0
    if mode == "by max": return [fi/m if m>0 else 0 for fi in f]
    return [fi/s if s>0 else 0 for fi in f]

def show_grid(img, gx, gy):
    w, h = img.size
    xs, ys = edges(w, h, gx, gy)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for x in xs[1:-1]: ax.axvline(x, linewidth=1)
    for y in ys[1:-1]: ax.axhline(y, linewidth=1)
    st.pyplot(fig)

def sign(u):
    return 1 if u>0 else -1

def perc_train(X, y, lr, epochs):
    n = len(X[0])
    w = np.zeros(n+1)
    hist = []
    for _ in range(epochs):
        idx = list(range(len(X)))
        random.shuffle(idx)
        errs = 0
        for k in idx:
            x = np.array([1.0]+X[k])
            yk = y[k]
            u = float(np.dot(w, x))
            yhat = sign(u)
            if yhat != yk:
                w += lr*(yk - yhat)*x
                errs += 1
        hist.append(errs)
        if errs==0: break
    return w, hist

def perc_pred(w, x):
    u = float(np.dot(w, np.array([1.0]+x)))
    return sign(u), u

with st.sidebar:
    gx = st.number_input("grid_x", 2, 20, 6)
    gy = st.number_input("grid_y", 2, 20, 6)
    thr = st.slider("threshold", 0, 255, 128)
    nmode = st.radio("normalization", ["by max","by sum"])
    lr = st.number_input("learning rate", 0.001, 1.0, 0.1, 0.001, format="%.3f")
    epochs = st.number_input("epochs", 1, 500, 50)

st.subheader("Training set (2 classes)")
c1, c2 = st.columns(2)
with c1:
    nameA = st.text_input("Class +1 name", value="Class_A")
    filesA = st.file_uploader("BMPs (≈10) for +1", type=["bmp"], accept_multiple_files=True, key="A")
with c2:
    nameB = st.text_input("Class -1 name", value="Class_B")
    filesB = st.file_uploader("BMPs (≈10) for -1", type=["bmp"], accept_multiple_files=True, key="B")

XA, XB = [], []
if filesA:
    st.write(f"{nameA}: {len(filesA)} samples")
    for f in filesA:
        img = Image.open(f)
        st.image(img, caption=f"{nameA}: {f.name}", width=140)
        show_grid(img, gx, gy)
        fa = feat_abs(img, gx, gy, thr)
        XA.append(feat_norm(fa, nmode))
if filesB:
    st.write(f"{nameB}: {len(filesB)} samples")
    for f in filesB:
        img = Image.open(f)
        st.image(img, caption=f"{nameB}: {f.name}", width=140)
        show_grid(img, gx, gy)
        fb = feat_abs(img, gx, gy, thr)
        XB.append(feat_norm(fb, nmode))

X, y = [], []
for v in XA: X.append(v); y.append(1)
for v in XB: X.append(v); y.append(-1)

st.divider()
st.subheader("Training")
w = None
if X and y:
    w, hist = perc_train(X, y, lr, epochs)
    st.write("errors per epoch:", " ".join(str(h) for h in hist[:50]) + (" ..." if len(hist)>50 else ""))
    st.write("weights (bias first):", " ".join(str(round(float(x),4)) for x in w[:min(12,len(w))]) + (" ..." if len(w)>12 else ""))
    if hist and hist[-1]==0: st.success("training converged (zero errors on last epoch)")
else:
    st.info("add training samples for both classes")

st.divider()
st.subheader("Classification of unknown")
unk = st.file_uploader("unknown BMP", type=["bmp"], key="U")
if unk and w is not None:
    uimg = Image.open(unk)
    st.image(uimg, caption="unknown", width=220)
    show_grid(uimg, gx, gy)
    fu = feat_norm(feat_abs(uimg, gx, gy, thr), nmode)
    st.write("unknown vector:", " ".join(str(round(x,3)) for x in fu[:min(12,len(fu))]) + (" ..." if len(fu)>12 else ""))
    yh, u = perc_pred(w, fu)
    cls = nameA if yh==1 else nameB
    st.write(f"raw score u = {round(u,6)}")
    st.success(f"class: {cls} (y={yh})")
elif unk and w is None:
    st.warning("train perceptron first")
