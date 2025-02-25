from fastapi import FastAPI, UploadFile, File
from models import predict_image
import utils

# Initialize FastAPI app with a title and description
app = FastAPI(
    title="Plant Disease Prediction API",
    description="API to predict plant diseases from images"
)

# Single API endpoint for prediction
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded image.

    Args:
        file: The image file to be uploaded for prediction.

    Returns:
        A JSON object with the prediction and disease information, or an error message if processing fails.
    """
    try:
        # Read the uploaded image as bytes
        img = file.file.read()
        # Get the prediction from the model
        prediction = predict_image(img)
        # Fetch disease information from the dictionary
        disease_info = utils.disease_dic[prediction]
        # Return prediction and disease info as JSON
        return {"prediction": prediction, "disease_info": disease_info}
    except Exception as e:
        # Return error message if something goes wrong
        return {"error": "Failed to process image"}