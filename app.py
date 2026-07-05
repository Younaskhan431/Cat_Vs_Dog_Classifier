import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
 
# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
 
st.set_page_config(
    page_title="Cat vs Dog AI",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# -------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------
 
MODEL_PATH = "model/best_fine_tuned_mobilenet.keras"
 
IMAGE_SIZE = (224,224)
 
CLASS_NAMES = [
    "Cat",
    "Dog"
]
 
# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
 
model = load_model()
 
# -------------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------------
 
def preprocess_image(image):
 
    image = image.convert("RGB")
 
    image = image.resize(IMAGE_SIZE)
 
    image = np.array(image,dtype=np.float32)
 
    image = np.expand_dims(image,axis=0)
 
    return image
 
# -------------------------------------------------------
# CSS
# -------------------------------------------------------
 
st.markdown("""
 
<style>
 
#MainMenu{
visibility:hidden;
}
 
footer{
visibility:hidden;
}
 
header{
visibility:hidden;
}
 
.stApp{
 
background:
linear-gradient(
135deg,
#EEF2FF,
#F8FAFC,
#FFFFFF
);
 
}
 
.block-container{
 
padding-top:2rem;
padding-bottom:2rem;
max-width:1400px;
 
}
 
.hero{
 
text-align:center;
 
padding:20px;
 
margin-bottom:30px;
 
}
 
.hero h1{
 
font-size:56px;
 
font-weight:800;
 
margin-bottom:10px;
 
background:linear-gradient(
90deg,
#4F46E5,
#06B6D4
);
 
-webkit-background-clip:text;
 
-webkit-text-fill-color:transparent;
 
}
 
.hero p{
 
font-size:19px;
 
color:#64748B;
 
}
 
.card{
 
background:white;
 
border-radius:25px;
 
padding:25px;
 
box-shadow:
0 15px 35px rgba(0,0,0,.08);
 
border:1px solid rgba(255,255,255,.6);
 
}
 
.prediction-card{
 
background:white;
 
border-radius:25px;
 
padding:30px;
 
text-align:center;
 
box-shadow:
0 20px 45px rgba(0,0,0,.10);
 
}
 
.result{
 
font-size:42px;
 
font-weight:800;
 
margin-top:15px;
 
color:#111827;
 
}
 
.confidence{
 
font-size:20px;
 
margin-top:10px;
 
color:#6B7280;
 
}
 
.small{
 
color:#94A3B8;
 
font-size:14px;
 
}
 
.metric-card{
 
background:#F8FAFC;
 
padding:20px;
 
border-radius:20px;
 
text-align:center;
 
}
 
.sidebar-title{
 
font-size:24px;
 
font-weight:700;
 
margin-bottom:10px;
 
}
 
</style>
 
""",unsafe_allow_html=True)
 
# -------------------------------------------------------
# HERO
# -------------------------------------------------------
 
st.markdown("""
 
<div class="hero">
 
<h1>🐾 Cat vs Dog AI Classifier</h1>
 
<p>
Upload an image and let Artificial Intelligence identify whether it is a Cat or Dog.
</p>
 
</div>
 
""",unsafe_allow_html=True)
 
# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
 
with st.sidebar:
 
    st.title("🤖 Model Information")
 
    st.divider()
 
    st.metric(
        "Architecture",
        "MobileNetV2"
    )
 
    st.metric(
        "Technique",
        "Fine-Tuning"
    )
 
    st.metric(
        "Framework",
        "TensorFlow"
    )
 
    st.metric(
        "Input Size",
        "224 × 224"
    )
 
    st.divider()
 
    st.success("Ready for Prediction")
 
    st.info(
        "Upload a clear image containing a single cat or dog."
    )
 
    st.divider()
 
    st.caption(
        "Developed by Younas Khan"
    )
 
# -------------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------------
 
left,right=st.columns([1.2,1],gap="large")
 
# -------------------------------------------------------
# LEFT COLUMN
# -------------------------------------------------------
 
with left:
 
    st.markdown('<div class="card">', unsafe_allow_html=True)
 
    st.subheader("📤 Upload Image")
 
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )
 
    if uploaded_file is None:
 
        st.markdown("""
 
        <br>
 
        <center>
 
        <h3 style="color:#64748B;">
        Drag & Drop an image here
        </h3>
 
        <p style="color:#94A3B8;">
        or click above to browse
        </p>
 
        </center>
 
        <br>
 
        """, unsafe_allow_html=True)
 
    else:
 
        image = Image.open(uploaded_file)
 
        st.image(
            image,
            caption="Uploaded Image",
            use_column_width=True
        )
 
        st.success("Image uploaded successfully.")
 
    st.markdown("</div>", unsafe_allow_html=True)
 
# -------------------------------------------------------
# RIGHT COLUMN
# -------------------------------------------------------
 
with right:
 
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
 
    st.subheader("🤖 AI Prediction")
 
    if uploaded_file is None:
 
        st.markdown("""
 
        <br><br>
 
        <center>
 
        <h2 style="color:#CBD5E1;">
        Waiting for Image...
        </h2>
 
        <p style="color:#94A3B8;">
        Upload a Cat or Dog image to start prediction.
        </p>
 
        </center>
 
        """, unsafe_allow_html=True)
 
    else:
 
        img = preprocess_image(image)
 
        with st.spinner("Analyzing image..."):
 
            progress = st.progress(0)
 
            for i in range(100):
 
                time.sleep(0.01)
 
                progress.progress(i + 1)
 
            prediction = model.predict(
                img,
                verbose=0
            )[0][0]
 
        progress.empty()
 
        if prediction > 0.5:
 
            predicted_class = "Dog"
 
            confidence = float(prediction)
 
            emoji = "🐶"
 
            color = "#2563EB"
 
        else:
 
            predicted_class = "Cat"
 
            confidence = float(1 - prediction)
 
            emoji = "🐱"
 
            color = "#EC4899"
 
        st.markdown(
            f"""
            <div style="font-size:90px;text-align:center;">
            {emoji}
            </div>
            """,
            unsafe_allow_html=True
        )
 
        st.markdown(
            f"""
            <div class="result" style="color:{color};text-align:center;">
            {predicted_class}
            </div>
            """,
            unsafe_allow_html=True
        )
 
        st.markdown(
            f"""
            <div class="confidence" style="text-align:center;">
            Confidence: <b>{confidence*100:.2f}%</b>
            </div>
            """,
            unsafe_allow_html=True
        )
 
        st.progress(confidence)
 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
 
        # -------------------------------------------------------
        # AI CONFIDENCE
        # -------------------------------------------------------
 
        st.subheader("📊 Prediction Confidence")
 
        if predicted_class == "Dog":
 
            dog_prob = confidence
            cat_prob = 1 - confidence
 
        else:
 
            cat_prob = confidence
            dog_prob = 1 - confidence
 
        col1, col2 = st.columns(2)
 
        with col1:
 
            st.metric(
                "🐱 Cat",
                f"{cat_prob*100:.2f}%"
            )
 
        with col2:
 
            st.metric(
                "🐶 Dog",
                f"{dog_prob*100:.2f}%"
            )
 
        st.write("")
 
        st.progress(max(cat_prob, dog_prob))
 
        st.write("")
 
        st.markdown("---")
 
        # -------------------------------------------------------
        # AI INSIGHTS
        # -------------------------------------------------------
 
        st.subheader("🧠 AI Insight")
 
        if confidence > 0.98:
 
            st.success(
                "The model is extremely confident about this prediction."
            )
 
        elif confidence > 0.90:
 
            st.info(
                "The prediction is highly reliable."
            )
 
        elif confidence > 0.75:
 
            st.warning(
                "The model is reasonably confident, but difficult images may reduce certainty."
            )
 
        else:
 
            st.error(
                "Low confidence prediction. Try uploading a clearer image."
            )
 
        st.markdown("</div>", unsafe_allow_html=True)
 
# -------------------------------------------------------
# MODEL DETAILS
# -------------------------------------------------------
 
st.write("")
 
st.markdown("## 🤖 Model Details")
 
a, b, c, d = st.columns(4)
 
with a:
 
    st.markdown("""
    <div class="metric-card">
 
    <h3>Architecture</h3>
 
    <h2>MobileNetV2</h2>
 
    </div>
    """, unsafe_allow_html=True)
 
with b:
 
    st.markdown("""
    <div class="metric-card">
 
    <h3>Technique</h3>
 
    <h2>Fine-Tuning</h2>
 
    </div>
    """, unsafe_allow_html=True)
 
with c:
 
    st.markdown("""
    <div class="metric-card">
 
    <h3>Framework</h3>
 
    <h2>TensorFlow</h2>
 
    </div>
    """, unsafe_allow_html=True)
 
with d:
 
    st.markdown("""
    <div class="metric-card">
 
    <h3>Input</h3>
 
    <h2>224 × 224</h2>
 
    </div>
    """, unsafe_allow_html=True)
 
st.write("")