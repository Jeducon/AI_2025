import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math

st.title("Лаба 3 — Адаптивне розпізнавання (порівняння з еталоном)")

grid_x = st.number_input("grid_x", 2, 20, 6)
grid_y = st.number_input("grid_y", 2, 20, 6)
thr = st.slider("Поріг бінаризації", 0, 255, 128)
metric = st.radio("Норма", ["Евклідова", "Манхетенська", "Чебишевська"])

st.subheader("1) Навчальна послідовність (класи)")
colA, colB, colC = st.columns(3)
with colA:
    clsA = st.text_input("Назва класу A", value="A")
    filesA = st.file_uploader("Зразки A (BMP)", type=["bmp"], accept_multiple_files=True, key="A")
with colB:
    clsB = st.text_input("Назва класу B", value="B")
    filesB = st.file_uploader("Зразки B (BMP)", type=["bmp"], accept_multiple_files=True, key="B")
with colC:
    clsC = st.text_input("Назва класу C", value="C")
    filesC = st.file_uploader("Зразки C (BMP)", type=["bmp"], accept_multiple_files=True, key="C")

def vec(img, gx, gy, t):
    img = img.convert("L")
    w, h = img.size
    xs = [round(i*w/gx) for i in range(gx+1)]
    ys = [round(j*h/gy) for j in range(gy+1)]
    a = np.array(img)
    f = []
    for i in range(gy):
        for j in range(gx):
            r = a[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            f.append(int(np.sum(r < t)))
    s = sum(f)
    m = max(f) if f else 0
    nsum = [fi/s if s>0 else 0 for fi in f]
    nmod = [fi/m if m>0 else 0 for fi in f]
    return f, nsum, nmod

def dist(x, y, kind):
    if kind=="Евклідова":
        return math.sqrt(sum((a-b)**2 for a,b in zip(x,y)))
    if kind=="Манхетенська":
        return sum(abs(a-b) for a,b in zip(x,y))
    return max(abs(a-b) for a,b in zip(x,y))

def show(img, gx, gy):
    w, h = img.size
    xs = [round(i*w/gx) for i in range(gx+1)]
    ys = [round(j*h/gy) for j in range(gy+1)]
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    for x in xs[1:-1]:
        ax.axvline(x, linewidth=1)
    for y in ys[1:-1]:
        ax.axhline(y, linewidth=1)
    st.pyplot(fig)

def proto(files):
    V = []
    for f in files or []:
        img = Image.open(f)
        _, _, v = vec(img, grid_x, grid_y, thr)
        V.append(v)
    if not V:
        return None
    M = np.mean(np.array(V), axis=0).tolist()
    return M

A = proto(filesA)
B = proto(filesB)
C = proto(filesC)

st.write("Еталони (середні нормовані вектори):")
if A: st.write(f"{clsA}:", " ".join(str(round(x,3)) for x in A[:min(10,len(A))]), " ...")
if B: st.write(f"{clsB}:", " ".join(str(round(x,3)) for x in B[:min(10,len(B))]), " ...")
if C: st.write(f"{clsC}:", " ".join(str(round(x,3)) for x in C[:min(10,len(C))]), " ...")

st.subheader("2) Класифікація невідомого образу")
unk = st.file_uploader("Невідомий BMP", type=["bmp"], key="U")
if unk and any([A,B,C]):
    uimg = Image.open(unk)
    st.image(uimg, caption="Невідомий образ", width=220)
    show(uimg, grid_x, grid_y)
    _, _, U = vec(uimg, grid_x, grid_y, thr)
    st.write("Нормований вектор невідомого:", " ".join(str(round(x,3)) for x in U))

    scores = []
    if A: scores.append((clsA, dist(U, A, metric)))
    if B: scores.append((clsB, dist(U, B, metric)))
    if C: scores.append((clsC, dist(U, C, metric)))
    scores.sort(key=lambda x: x[1])
    for k,dv in scores:
        st.write(f"{k}: {round(dv,4)}")
    st.success(f"Клас: {scores[0][0]}")
elif unk and not any([A,B,C]):
    st.warning("Спочатку додай навчальні зразки хоча б одного класу.")
