import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title("Lab 5 — Hopfield Network (3 classes)")

def split_edges(w,h,gx,gy):
    xs=[round(i*w/gx) for i in range(gx+1)]
    ys=[round(j*h/gy) for j in range(gy+1)]
    return xs,ys

def feat_abs(img,gx,gy,thr):
    a=np.array(img.convert("L"))
    h,w=a.shape
    xs,ys=split_edges(w,h,gx,gy)
    f=[]
    for i in range(gy):
        for j in range(gx):
            r=a[ys[i]:ys[i+1],xs[j]:xs[j+1]]
            f.append(int(np.sum(r<thr)))
    return np.array(f,dtype=float)

def feat_bipolar(img,gx,gy,thr,cell_ratio):
    f=feat_abs(img,gx,gy,thr)
    cell_size=(img.size[0]//gx)*(img.size[1]//gy)
    b=(f>cell_ratio*cell_size).astype(int)
    b=np.where(b==1,1,-1).astype(float)
    return b

def show_grid(img,gx,gy):
    w,h=img.size
    xs,ys=split_edges(w,h,gx,gy)
    fig,ax=plt.subplots()
    ax.imshow(img,cmap="gray")
    for x in xs[1:-1]:
        ax.axvline(x,linewidth=1)
    for y in ys[1:-1]:
        ax.axhline(y,linewidth=1)
    st.pyplot(fig)

def hopfield_weights(patterns):
    n=patterns[0].size
    W=np.zeros((n,n),dtype=float)
    for x in patterns:
        W+=np.outer(x,x)
    np.fill_diagonal(W,0.0)
    return W

def sign_vec(v):
    y=np.where(v>0,1,-1).astype(float)
    return y

def iterate_sync(W,y0,max_iter=50):
    y=y0.copy()
    hist=[y.copy()]
    for _ in range(max_iter):
        y_new=sign_vec(W@y)
        hist.append(y_new.copy())
        if np.array_equal(y_new,y):
            break
        y=y_new
    return y,hist

def hamming(a,b):
    return int(np.sum(a!=b))

with st.sidebar:
    gx=st.number_input("grid_x",2,20,10)
    gy=st.number_input("grid_y",2,20,10)
    thr=st.slider("gray threshold",0,255,150)
    cell_ratio=st.slider("cell black ratio",0.0,1.0,0.05,0.01)
    max_iter=st.number_input("max iters",1,200,50)
    show_W=st.checkbox("show weight matrix",False)

st.subheader("1) Reference patterns (3 classes, multiple BMP each)")
cols=st.columns(3)
names=[]; files_all=[]; patterns=[]; labels=[]
for ci,c in enumerate(cols):
    with c:
        name=st.text_input(f"class {ci+1} name",value=f"class_{ci+1}",key=f"name_{ci}")
        names.append(name)
        fs=st.file_uploader("BMPs",type=["bmp"],accept_multiple_files=True,key=f"files_{ci}")
        files_all.append(fs)
        if fs:
            st.write(f"{name}: {len(fs)} samples")
            first_etalon=True
            for f in fs:
                img=Image.open(f)

                img = img.resize((200, 200))

                st.image(img,caption=f.name,use_container_width=False,width=140)
                show_grid(img,gx,gy)
                f_abs=feat_abs(img,gx,gy,thr)
                x=feat_bipolar(img,gx,gy,thr,cell_ratio)
                with st.expander(f"{name} features: {f.name}"):
                    st.write("absolute features:")
                    st.text(" ".join(str(int(v)) for v in f_abs))

                    st.write("bipolar features:")
                    st.text(" ".join(str(int(v)) for v in x))
                if first_etalon:
                    patterns.append(x)
                    labels.append(name)
                    first_etalon=False

st.divider()
st.subheader("2) Train Hopfield (build W)")

W=None
if patterns:
    m=len(patterns); n=patterns[0].size
    if m>int(0.15*n):
        st.warning(f"Capacity warning: stored {m} patterns, ~0.15·n ≈ {int(0.15*n)}")
    W=hopfield_weights(patterns)
    st.write(f"W shape: {W.shape[0]}×{W.shape[1]}")
    if show_W:
        with st.expander("W (top-left 20×20)"):
            tl=min(20,W.shape[0])
            st.dataframe(np.array(W[:tl,:tl]))
        with st.expander("stored patterns (rows)"):
            st.dataframe(np.vstack(patterns).astype(int))
else:
    st.info("Upload samples above to form stored patterns.")

st.divider()
st.subheader("3) Classify unknown (Hopfield retrieval)")

unk=st.file_uploader("unknown BMP",type=["bmp"],key="unknown")
if unk and W is not None:
    uimg=Image.open(unk)

    uimg = uimg.resize((200, 200))
    st.image(uimg,caption="unknown",width=220)
    show_grid(uimg,gx,gy)
    f_abs_unk=feat_abs(uimg,gx,gy,thr)
    y0=feat_bipolar(uimg,gx,gy,thr,cell_ratio)
    with st.expander("unknown features"):
        st.write("absolute features:")
        st.text(" ".join(str(int(v)) for v in f_abs_unk))

        st.write("bipolar features:")
        st.text(" ".join(str(int(v)) for v in y0))

    yF,hist=iterate_sync(W,y0,max_iter=max_iter)
    st.write("iterations:",len(hist)-1)
    with st.expander("states (first 50 dims)"):
        lines=[]
        for t,yy in enumerate(hist):
            s = " ".join(str(int(v)) for v in yy)
            lines.append(f"t={t}: {s}")
        st.text("\n".join(lines))
    dists=[]
    for k in range(len(patterns)):
        d=min(hamming(yF,patterns[k]),hamming(-yF,patterns[k]))
        dists.append((labels[k],d))
    dists.sort(key=lambda z:z[1])
    st.write("Hamming to stored patterns (by class):")
    for nm,dd in dists:
        st.write(f"{nm}: {dd}")
    best_class=dists[0][0]
    st.success(f"class: {best_class}")

elif unk and W is None:
    st.warning("Build W first: add reference patterns.")



