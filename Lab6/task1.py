import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title("Lab 6")

def split_edges(w,h,gx,gy):
    xs=[round(i*w/gx) for i in range(gx+1)]
    ys=[round(j*h/gy) for j in range(gy+1)]
    return xs,ys

def vec_bin(img,gx,gy,thr,ratio):
    a=np.array(img.convert("L"))
    h,w=a.shape
    xs,ys=split_edges(w,h,gx,gy)
    f=[]
    for i in range(gy):
        for j in range(gx):
            r=a[ys[i]:ys[i+1],xs[j]:xs[j+1]]
            cell=np.sum(r<thr)
            f.append(1 if cell>ratio*r.size else 0)
    return np.array(f,dtype=float)

def show_grid(img,gx,gy):
    w,h=img.size
    xs,ys=split_edges(w,h,gx,gy)
    fig,ax=plt.subplots()
    ax.imshow(img,cmap="gray")
    for x in xs[1:-1]: ax.axvline(x,linewidth=1)
    for y in ys[1:-1]: ax.axhline(y,linewidth=1)
    st.pyplot(fig)

def majority_proto(X):
    if not X: return None
    A=np.stack(X,axis=0)
    m=A.mean(axis=0)
    return (m>=0.5).astype(float)

def layer1_scores(x, protos):
    n=len(x)
    B=n/2.0
    S=[]
    for p in protos:
        y=0.5*np.dot(x,p)+B
        S.append(float(y))
    return np.array(S,dtype=float)

def hamming_wta(y, v, max_iter):
    y=np.maximum(0.0,y.astype(float))
    hist=[y.copy()]
    m=len(y)
    for _ in range(max_iter):
        sum_all=np.sum(y)
        inhib=v*(sum_all - y)
        y_new=np.maximum(0.0, y - inhib)
        hist.append(y_new.copy())
        if np.allclose(y_new,y,atol=1e-9): break
        y=y_new
    return y,hist

with st.sidebar:
    gx=st.number_input("grid_x",2,20,6)
    gy=st.number_input("grid_y",2,20,6)
    thr=st.slider("gray threshold",0,255,128)
    ratio=st.slider("cell black ratio",0.0,1.0,0.30,0.01)
    k=st.number_input("num classes",2,12,4)
    v=st.number_input("v (0..1/m)",0.001,0.5,0.05,0.001,format="%.3f")
    max_iter=st.number_input("max iters",1,200,50)
    show_s=st.checkbox("show L1 scores",False)

st.subheader("1) Training examples")
tabs=st.tabs([f"Клас {i+1}" for i in range(k)])
names=[]; protos=[]
for i,t in enumerate(tabs):
    with t:
        cname=st.text_input("Class name",value=f"class_{i+1}",key=f"name_{i}")
        names.append(cname)
        fs=st.file_uploader("Bmp examples",type=["bmp"],accept_multiple_files=True,key=f"files_{i}")
        X=[]
        if fs:
            st.write(f"{cname}: {len(fs)} examples")
            for f in fs:
                img=Image.open(f)
                st.image(img,caption=f.name,width=140)
                show_grid(img,gx,gy)
                X.append(vec_bin(img,gx,gy,thr,ratio))
        P=majority_proto(X)
        if P is not None:
            protos.append(P)
            st.write("first 48 bits", " ".join(str(int(b)) for b in P[:min(48,len(P))]))
        else:
            protos.append(None)
            st.write("Prototype: —")

st.divider()
st.subheader("2) Classification with Hamming")

unk=st.file_uploader("Unknown BMP",type=["bmp"],key="unknown")
if unk and any(p is not None for p in protos):
    uimg=Image.open(unk)
    st.image(uimg,caption="unknown",width=220)
    show_grid(uimg,gx,gy)
    x=vec_bin(uimg,gx,gy,thr,ratio)
    P=[p for p in protos if p is not None]; N=[n for n,p in zip(names,protos) if p is not None]
    y1=layer1_scores(x,P)
    if show_s:
        st.write("Layer-1 scores:", " ".join(str(round(s,3)) for s in y1))
    m=len(P)
    vmax=1.0/max(1,m)
    vv=min(v, vmax-1e-6) if m>0 else v
    yF,hist=hamming_wta(y1, vv, max_iter)
    st.write("Iterations:", len(hist)-1)
    with st.expander("Exit (top-50):"):
        st.text("t=0:  " + " ".join(str(round(s,3)) for s in hist[0][:min(50,len(hist[0]))]))
        if len(hist)>1:
            st.text("t=end:" + " ".join(str(round(s,3)) for s in hist[-1][:min(50,len(hist[-1]))]))
    j=int(np.argmax(yF))
    st.success(f"Class: {N[j]}")
elif unk:
    st.warning("Add examples for all classes.")
