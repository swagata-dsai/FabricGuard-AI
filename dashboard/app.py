import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "fabricguard_cnn_v2.keras"

IMG_SIZE = 128
PATCH_SIZE = 256
THRESHOLD = 0.5

st.set_page_config(
    page_title="FabricGuard AI",
    page_icon="🧵",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def create_patches(image):
    image = image.convert("RGB")
    width, height = image.size
    patches = []

    for y in range(0, height, PATCH_SIZE):
        for x in range(0, width, PATCH_SIZE):
            patch = image.crop((
                x, y,
                min(x + PATCH_SIZE, width),
                min(y + PATCH_SIZE, height)
            ))

            if patch.size != (PATCH_SIZE, PATCH_SIZE):
                canvas = Image.new("RGB", (PATCH_SIZE, PATCH_SIZE))
                canvas.paste(patch, (0, 0))
                patch = canvas

            patches.append(patch)

    return patches

st.title("🧵 FabricGuard AI")
st.subheader("Intelligent Textile Defect Detection & Quality Assessment")

st.info(
    "Upload a fabric image to receive patch-level defect predictions. "
    "This prototype performs image classification, not pixel-level segmentation."
)

with st.sidebar:
    st.header("Model Information")
    st.write("Model: CNN V2")
    st.write("Input size: 128 × 128")
    st.write("Patch size: 256 × 256")
    st.write("Classification threshold: 0.50")
    st.divider()
    st.caption("Academic prototype using the AITEX Fabric Image Database.")

uploaded_file = st.file_uploader(
    "Upload Fabric Image",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Fabric Image", use_container_width=True)
    st.write(f"Image dimensions: {image.width} × {image.height}")

    if st.button("Analyze Fabric", type="primary"):
        try:
            model = load_model()
            patches = create_patches(image)

            batch = np.array([
                np.array(patch.resize((IMG_SIZE, IMG_SIZE)))
                for patch in patches
            ], dtype=np.float32) / 255.0

            with st.spinner("Analyzing fabric patches..."):
                probabilities = model.predict(batch, verbose=0).flatten()

            results = []

            for index, probability in enumerate(probabilities):
                prediction = "Defect" if probability >= THRESHOLD else "Normal"

                results.append({
                    "Patch": index + 1,
                    "Defect Probability (%)": round(float(probability) * 100, 2),
                    "Prediction": prediction
                })

            df = pd.DataFrame(results)

            total = len(df)
            defect_count = int((df["Prediction"] == "Defect").sum())
            normal_count = total - defect_count

            st.divider()
            st.header("Quality Assessment Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Patches", total)
            col2.metric("Predicted Normal", normal_count)
            col3.metric("Predicted Defect", defect_count)

            if defect_count > 0:
                st.warning(
                    f"{defect_count} patch(es) flagged for possible defects. "
                    "Manual inspection is recommended."
                )
            else:
                st.success("No patches were flagged by the model.")

            st.subheader("Patch-wise Prediction")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.subheader("Defect Probability by Patch")
            chart_data = df.set_index("Patch")["Defect Probability (%)"]
            st.bar_chart(chart_data)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Report",
                data=csv,
                file_name="fabricguard_prediction_report.csv",
                mime="text/csv"
            )

            st.caption(
                "Important: These predictions are experimental. "
                "CNN V2 achieved 11.31% PR-AUC and 44.83% defect recall "
                "on the held-out test set. Results should not be treated "
                "as a reliable industrial quality inspection."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

else:
    st.info("Please upload a fabric image to begin analysis.")

st.divider()
st.caption("FabricGuard AI | MBA Data Science & Artificial Intelligence Project")
