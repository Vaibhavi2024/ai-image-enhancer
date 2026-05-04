import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Image Enhancer", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1 {
    text-align: center;
    color: white;
    font-size: 48px;
}
.stSlider label, .stCheckbox label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI Image Enhancement Studio</h1>", unsafe_allow_html=True)

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- CONTROLS ----------
    st.sidebar.header("⚙️ AI Controls")

    brightness = st.sidebar.slider("Brightness", -50, 50, 0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
    sharpness = st.sidebar.slider("Sharpness", 0, 3, 1)

    auto_mode = st.sidebar.checkbox("🤖 Auto Enhance")

    # ---------- PROCESS ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    # Apply brightness & contrast
    enhanced = cv2.convertScaleAbs(equalized, alpha=contrast, beta=brightness)

    # Apply sharpening
    if sharpness > 0:
        kernel = np.array([[0, -1, 0],
                           [-1, 5 + sharpness, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

    # Auto mode override
    if auto_mode:
        enhanced = cv2.equalizeHist(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=10)

    # ---------- SLIDER VIEW ----------
    st.subheader("🎚️ Before vs After")

    slider = st.slider("Compare", 0, 100, 50)
    blended = cv2.addWeighted(gray, slider/100, enhanced, 1-slider/100, 0)

    st.image(blended, use_column_width=True)

    # ---------- IMAGE DISPLAY ----------
    st.subheader("🖼️ Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_rgb, caption="Original")

    with col2:
        st.image(gray, caption="Grayscale")

    with col3:
        st.image(enhanced, caption="AI Enhanced")

    # ---------- HISTOGRAM ----------
    st.subheader("📊 Histogram")

    fig, ax = plt.subplots(1,2)

    ax[0].hist(gray.ravel(), bins=256)
    ax[0].set_title("Before")

    ax[1].hist(enhanced.ravel(), bins=256)
    ax[1].set_title("After")

    st.pyplot(fig)

    # ---------- DOWNLOAD ----------
    st.download_button(
        "📥 Download Image",
        data=cv2.imencode('.jpg', enhanced)[1].tobytes(),
        file_name="ai_enhanced.jpg",
        mime="image/jpeg"
    )