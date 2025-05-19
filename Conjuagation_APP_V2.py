import streamlit as st
import numpy as np
import onnxruntime as ort
import tempfile
import os
import requests

# 🔗 Map display names to GitHub raw URLs
MODEL_OPTIONS = {
    "2L to 5L": "https://github.com/code2mech/App/blob/main/2L_5L.onnx",
    "10L to 20L": "https://github.com/code2mech/App/blob/main/10L_20L.onnx",
    "800L to 2000L": "https://raw.githubusercontent.com/yourusername/yourrepo/main/model_heat.onnx"
}

# 🏷 Custom output labels
OUTPUT_LABELS = ["Shear Rate", "Power", "Tip Speed", "Reynolds Number", "Power Number"]

def download_model_from_github(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name
    else:
        raise Exception("Failed to download model from GitHub.")

def main():
    st.set_page_config(page_title="ONNX Model Inference", layout="wide")
    st.title("🧠 ONNX Model Inference from GitHub")

    # 👇 Dropdown to choose model
    selected_model_name = st.selectbox("Select a model", list(MODEL_OPTIONS.keys()))
    selected_model_url = MODEL_OPTIONS[selected_model_name]

    volume = st.number_input("Volume", value=1.0)
    impeller_speed = st.number_input("Impeller Speed", value=1.0)

    if st.button("Run Inference"):
        with st.spinner(f"Running inference with {selected_model_name}..."):
            try:
                model_path = download_model_from_github(selected_model_url)
                success, result = run_inference(model_path, volume, impeller_speed)

                if success:
                    _, _, output_names, outputs = result
                    output_array = outputs[0].flatten()

                    st.subheader("📊 Results")
                    if len(output_array) >= 5:
                        for i in range(5):
                            st.write(f"**{OUTPUT_LABELS[i]}**: {output_array[i]}")
                    else:
                        st.warning("Not enough outputs to label. Showing raw values:")
                        for i, val in enumerate(output_array):
                            st.write(f"Output {i + 1}: {val}")

                    # CSV Download
                    st.download_button(
                        label="Download results as CSV",
                        data=",".join(map(str, output_array)),
                        file_name="inference_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Inference failed: {result}")
            finally:
                if os.path.exists(model_path):
                    os.remove(model_path)

def run_inference(model_path, volume, impeller_speed):
    try:
        session = ort.InferenceSession(model_path)
        input_names = [i.name for i in session.get_inputs()]
        input_data = np.array([[volume, impeller_speed]], dtype=np.float32)
        input_dict = {input_names[0]: input_data}
        outputs = session.run(None, input_dict)
        return True, (input_names, input_dict, session.get_outputs(), outputs)
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    main()
