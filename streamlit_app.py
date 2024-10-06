import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
from keplergl import KeplerGl

# Load the pre-trained model and processor from Hugging Face
@st.cache_resource
def load_model():
    model = AutoModelForSemanticSegmentation.from_pretrained("sawthiha/segformer-b0-finetuned-deprem-satellite")
    processor = AutoImageProcessor.from_pretrained("sawthiha/segformer-b0-finetuned-deprem-satellite")
    return model, processor

# Preprocess the uploaded image
def preprocess_image(image, processor):
    inputs = processor(images=image, return_tensors="pt")
    return inputs

# Make predictions on the image
def predict_image(image, model, processor):
    inputs = preprocess_image(image, processor)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the output to get segmentation mask
    predicted_mask = torch.argmax(outputs.logits, dim=1).cpu().numpy()
    return predicted_mask[0]

# Kepler visualization function
def visualize_with_kepler(predicted_mask):
    # Convert the segmentation mask to a geojson-like format for visualization
    features = []
    for row in range(predicted_mask.shape[0]):
        for col in range(predicted_mask.shape[1]):
            # Example data transformation: Each pixel can be mapped to a lat/long and assigned the predicted class
            features.append({
                "type": "Feature",
                "properties": {"class": int(predicted_mask[row][col])},
                "geometry": {
                    "type": "Point",
                    "coordinates": [123.89 + col * 0.001, 10.32 + row * 0.001]  # Adjust lat/long to fit map area
                }
            })
    
    # Create geojson data for Kepler.gl
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # Kepler.gl configuration
    config = {
        "version": "v1",
        "config": {
            "mapState": {
                "bearing": 0,
                "latitude": 10.3157,  # Adjust latitude
                "longitude": 123.8854,  # Adjust longitude
                "pitch": 0,
                "zoom": 10  # Adjust zoom level
            },
            "visState": {
                "layers": [
                    {
                        "type": "geojson",
                        "config": {
                            "dataId": "segmentation_data",
                            "label": "Segmentation Map",
                            "color": [0, 255, 0],  # Green for the mask
                            "columns": {"geojson": "geometry"},
                        }
                    }
                ]
            }
        }
    }

    # Load Kepler.gl with the data and configuration
    kepler_map = KeplerGl(data={'segmentation_data': geojson_data}, config=config)
    return kepler_map

# Streamlit App Interface
def main():
    st.title("Satellite Image Segmentation with SegFormer and Kepler.gl")
    
    # Upload image section
    uploaded_image = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Load the uploaded image
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load model and processor
        model, processor = load_model()

        # Run prediction on uploaded image
        if st.button("Segment and Visualize"):
            prediction = predict_image(image, model, processor)
            st.write("Segmentation complete!")
            
            # Visualize segmentation mask with Kepler.gl
            kepler_map = visualize_with_kepler(prediction)
            kepler_map.save_to_html(file_name='kepler_map.html')
            
            # Embed the kepler.gl map in the streamlit app
            st.components.v1.html(open("kepler_map.html", "r").read(), height=500)

if __name__ == '__main__':
    main()
