import streamlit as st
import numpy as np
import cv2
import joblib
import os
import plotly.express as px
from tensorflow.keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

# --- 1. إعدادات الصفحة الأساسية ---
st.set_page_config(
    page_title="BIO-GUARD AI| Biometric Terminal",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --- 2. حقن التصميم الاحترافي (نسخة مضغوطة للشاشة الواحدة) ---
def apply_terminal_theme():
    terminal_css = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@300;500;700&display=swap');

        /* تقليل الهوامش العلوية والسفلية للمحتوى */
        .block-container { padding-top: 1rem !important; padding-bottom: 0rem !important; }
        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background-color: #0d1117 !important;
            color: #c9d1d9 !important;
            font-family: 'Inter', sans-serif !important;
        }

        .logo-container {
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            background: linear-gradient(90deg, #58a6ff, #3fb950);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem; /* تصغير اللوجو قليلاً */
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 0px;
        }

        .verdict-card {
            background: linear-gradient(145deg, #161b22, #0d1117);
            padding: 20px; /* تقليل الحشو الداخلي */
            border-radius: 20px;
            border: 1px solid #30363d;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            margin-top: 10px; /* تقليل المسافة العلوية */
        }

        .pattern-highlight {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem !important; /* تصغير حجم خط النمط */
            font-weight: 700;
            color: #58a6ff;
            text-shadow: 0 0 10px rgba(88, 166, 255, 0.4);
            letter-spacing: 1px;
            margin-bottom: 10px;
            display: block;
        }

        div.stButton > button {
            width: 100% !important;
            background-color: #238636 !important;
            color: white !important;
            border: none !important;
            height: 3.5rem !important; /* تقليل ارتفاع الزر */
            font-weight: 700 !important;
            border-radius: 12px !important;
            font-size: 1rem !important;
            text-transform: uppercase !important;
            margin-top: 10px !important;
        }

        .status-badge {
            padding: 5px 15px;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 700;
            background: #161b22;
            border: 1px solid #30363d;
            color: #58a6ff;
        }

        .waiting-box {
            text-align: center;
            padding: 50px 20px; /* تقليل الطول الكلي لصندوق الانتظار */
            background: #0d1117;
            border-radius: 20px;
            border: 2px dashed #30363d;
            color: #8b949e;
            margin-top: 15px;
        }

        /* تحسين منطقة رفع الملفات لتكون مدمجة */
        .stFileUploader section { padding: 0.5rem 1rem !important; }
    </style>
    """
    st.markdown(terminal_css, unsafe_allow_html=True)


# --- 3. محرك تحميل البيانات (بدون تغيير) ---
@st.cache_resource
def boot_system():
    try:
        base = os.path.dirname(__file__)

        def p(f):
            return os.path.join(base, 'models', f)

        clf = load_model(p('class_model_f0.h5'))
        return {
            'clf': clf,
            'ext': Model(inputs=clf.inputs, outputs=clf.layers[6].output),
            'le': joblib.load(p('label_encoder.pkl')),
            'pca': joblib.load(p('pca_model.pkl')),
            'X': np.load(p('X_train_pca.npy')),
            'Yp': np.load(p('Y_train_person.npy')),
            'Yc': np.load(p('Y_train_class.npy'))
        }
    except:
        return None


def process_frame(raw_bytes):
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    enhanced = cv2.createCLAHE(4.0, (8, 8)).apply(img)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary.reshape(1, 128, 128, 1).astype('float32') / 255.0, binary


# --- 4. تشغيل الواجهة ---
def main():
    apply_terminal_theme()

    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown('<div class="logo-container"><i class="fas fa-fingerprint"></i> BIO-GUARD AI</div>',
                    unsafe_allow_html=True)
    with h_right:
        st.markdown("""
            <div style="text-align: right; padding-top: 5px;">
                <span style="color:#3fb950; font-weight:bold; font-size:0.75rem;">● NODE_ACTIVE</span><br>
                <div class="status-badge" style="margin-top:5px;"><i class="fas fa-shield-virus"></i> SECURE TERMINAL</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.5rem 0; border-color:#30363d;'>", unsafe_allow_html=True)

    engine = boot_system()
    if not engine:
        st.error("SYSTEM ERROR: Resources not found in /models/ directory.")
        return

    left_side, right_side = st.columns([1, 1.2], gap="large")

    with left_side:
        st.markdown("##### <i class='fas fa-microscope'></i> Scanned Input", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop fingerprint", type=['png', 'jpg', 'tif', 'jpeg'],
                                         label_visibility="collapsed")

        if uploaded_file:
            tensor, preview = process_frame(uploaded_file.read())
            # تحديد عرض الصورة لمنعها من دفع العناصر للأسفل
            st.image(preview, width=250)
            trigger = st.button("🚀 START SCAN")
        else:
            trigger = False

    with right_side:
        st.markdown("##### <i class='fas fa-project-diagram'></i> Analysis Output", unsafe_allow_html=True)

        if uploaded_file and trigger:
            with st.spinner("Decoding..."):
                raw_out = engine['clf'].predict(tensor, verbose=0)[0]
                pred_cls = engine['le'].inverse_transform([np.argmax(raw_out)])[0].upper()

                feat = engine['ext'].predict(tensor, verbose=0).reshape(1, -1)
                norm = Normalizer(norm='l2')
                pca_feat = engine['pca'].transform(norm.transform(feat))
                probe = norm.transform(pca_feat)

                mask = (engine['Yc'] == pred_cls)
                if np.any(mask):
                    sims = cosine_similarity(probe, engine['X'][mask])[0]
                    best = np.argmax(sims)
                    score = sims[best]
                    user = engine['Yp'][mask][best]

                    st.markdown(f"""
                        <div class="verdict-card">
                            <div class="pattern-highlight"><i class="fas fa-dna"></i> PATTERN: {pred_cls}</div>
                            <div class="status-badge" style="color:#3fb950; border-color:#3fb950; margin-bottom:15px;">IDENTITY VERIFIED</div>
                            <br><small style="color:#8b949e;">IDENTIFIED AS:</small>
                            <h1 style="font-family:'Orbitron'; color:white; font-size:2.8rem; margin:5px 0;">USER #{user}</h1>
                            <div style="margin-top:15px;">
                                <div style="display:flex; justify-content:space-between; font-size:0.75rem; margin-bottom:5px;">
                                    <span>Reliability</span>
                                    <span style="color:#58a6ff;">{score:.4f}</span>
                                </div>
                                <div style="background:#30363d; height:8px; border-radius:10px;">
                                    <div style="background:linear-gradient(90deg, #58a6ff, #3fb950); width:{score * 100}%; height:100%; border-radius:10px;"></div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No matching subject found.")
        else:
            st.markdown("""
                <div class="waiting-box">
                    <i class="fas fa-fingerprint fa-2x" style="opacity:0.2; margin-bottom:10px;"></i><br>
                    <span style="font-size:0.8rem;">READY FOR SIGNAL</span>
                </div>
            """, unsafe_allow_html=True)

    # فوتر مدمج جداً
    st.markdown(
        f"<div style='text-align:center; margin-top:30px; color:#484f58; font-size:0.7rem;'>BIO-CORE v2.4 • 2026</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()