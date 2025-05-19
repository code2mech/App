import streamlit as st
import numpy as np
import onnxruntime as ort
import tempfile
import os
import io

def main():
    st.set_page_config(
        page_title="ONNX Model Inference",
        page_icon="🧠",
        layout="wide"
    )

    st.title("ONNX Model Inference Tool")
    st.write("Upload your ONNX model and run inference in the cloud!")

    # Create sidebar with information
    with st.sidebar:
        st.header("About")
        st.info(
            "This app allows you to upload an ONNX model and run inference with custom input values. "
            "The model should accept two input values: Volume and Impeller Speed."
        )
        
        st.header("Instructions")
        st.markdown(
            """
            1. Upload your ONNX model using the file uploader
            2. Enter the required input values
            3. Click 'Run Inference' to see the results
            """
        )

    # File uploader for the ONNX model
    uploaded_file = st.file_uploader("Upload your ONNX model", type=["onnx"])

    # Input fields for the two required values
    st.subheader("Model Inputs")
    col1, col2 = st.columns(2)
    with col1:
        volume = st.number_input("Volume", value=1.0, format="%.6f")
    with col2:
        impeller_speed = st.number_input("Impeller Speed", value=1.0, format="%.6f")

    # Run inference when button is clicked
    if uploaded_file is not None:
        if st.button("Run Inference"):
            with st.spinner("Running inference..."):
                # Get the bytes from the uploaded file
                model_bytes = uploaded_file.getvalue()
                
                # Run inference
                success, result = run_inference(model_bytes, volume, impeller_speed)
                
                if success:
                    input_names, input_dict, output_names, outputs = result
                    
                    # Display input information
                    st.subheader("Model Information")
                    st.write(f"Number of inputs: {len(input_names)}")
                    st.write(f"Input names: {input_names}")
                    
                    # Display input values
                    st.subheader("Input Values")
                    for name, value in input_dict.items():
                        st.write(f"Input '{name}':")
                        st.write(f"  Shape: {value.shape}")
                        st.write(f"  Values: {value}")
                    
                    # Display output information
                    st.subheader("Results")
                    for i, output_name in enumerate(output_names):
                        st.write(f"Output '{output_name}':")
                        st.write(f"  Shape: {outputs[i].shape}")
                        
                        # Display the output values in a more readable format
                        if outputs[i].size < 10:  # If small output, show directly
                            st.write(f"  Values: {outputs[i]}")
                        else:  # For larger outputs, use a dataframe
                            output_flat = outputs[i].flatten()
                            st.write("  First 10 values:")
                            st.write(output_flat[:10])
                            
                            # Option to download full results
                            output_csv = io.StringIO()
                            np.savetxt(output_csv, output_flat, delimiter=',')
                            st.download_button(
                                label="Download full results as CSV",
                                data=output_csv.getvalue(),
                                file_name=f"{output_name}_results.csv",
                                mime="text/csv"
                            )
                else:
                    st.error(f"Error running inference: {result}")
    else:
        st.info("Please upload an ONNX model to begin.")

    # Add footer
    st.markdown("---")
    st.caption("ONNX Model Inference Tool - Powered by Streamlit and ONNX Runtime")

def run_inference(model_bytes, volume, impeller_speed):
    try:
        # Create a temporary file to save the uploaded model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
            tmp_file.write(model_bytes)
            model_path = tmp_file.name
        
        # Load the ONNX model
        session = ort.InferenceSession(model_path)
        
        # Get input details
        inputs = session.get_inputs()
        input_names = [input.name for input in inputs]
        
        # Create input dictionary
        input_dict = {}
        
        for input_info in inputs:
            name = input_info.name
            shape = input_info.shape
            
            # Create input array based on shape
            values = [volume, impeller_speed]
            
            if len(shape) == 1:
                # 1D array (e.g., [2])
                input_data = np.array(values, dtype=np.float32)
            elif len(shape) == 2:
                # 2D array (e.g., [1, 2])
                input_data = np.array([values], dtype=np.float32)
            else:
                return False, f"Unexpected input shape: {shape}. Model should accept 1D or 2D input."
            
            input_dict[name] = input_data
        
        # Get output names
        output_names = [output.name for output in session.get_outputs()]
        
        # Run inference
        outputs = session.run(output_names, input_dict)
        
        # Clean up the temporary file
        os.unlink(model_path)
        
        return True, (input_names, input_dict, output_names, outputs)
    
    except Exception as e:
        # Clean up if possible
        try:
            if 'model_path' in locals():
                os.unlink(model_path)
        except:
            pass
        return False, str(e)

if __name__ == "__main__":
    main()