# AI-Based Cattle & Buffalo Breed Identification System

A web application for identifying Indian cattle and buffalo breeds using AI image classification.

## Features

- 📷 **Image Upload**: Drag-and-drop or click to upload images
- 🔍 **Breed Detection**: AI-powered breed identification
- 📊 **Confidence Scores**: Shows prediction confidence and top alternatives
- 💡 **Explainable AI**: Provides reasoning for breed predictions
- 📱 **Responsive Design**: Works on desktop and mobile devices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Using Your Own Model

### Option 1: Replace Demo Function

Edit `app.py` and replace the `predict_breed_demo()` function with your model:

```python
def predict_breed(image):
    # Load your model
    model = load_model('models/your_model.h5')
    
    # Preprocess image
    preprocessed = preprocess_image(image)
    
    # Get predictions
    predictions = model.predict(preprocessed)
    
    # Process and return results
    breed_names = ['Gir', 'Sahiwal', 'Murrah', ...]  # Your breed list
    top_indices = np.argsort(predictions[0])[::-1][:5]
    
    results = []
    for idx in top_indices:
        results.append({
            'breed': breed_names[idx],
            'confidence': float(predictions[0][idx])
        })
    
    return results
```

### Option 2: Train Your Own Model

1. **Prepare your dataset**:
   - Organize images by breed in folders
   - Ensure balanced dataset (recommended: 500+ images per breed)
   - Include various angles: face, side body, horns, hump

2. **Train a model**:
   ```python
   # Example training script
   from tensorflow import keras
   from tensorflow.keras import layers
   
   # Load and preprocess data
   # Train model
   # Save model
   model.save('models/breed_classifier.h5')
   ```

3. **Update `app.py`**:
   - Uncomment model loading code
   - Update `preprocess_image()` to match your model's input requirements
   - Replace `predict_breed_demo()` with actual model inference

## Dataset Sources

You can use datasets from:
- Government livestock databases
- University research datasets
- Field-collected images
- Online repositories (Kaggle, etc.)

## Supported Breeds

Currently configured for:
- **Cattle**: Gir, Sahiwal, Red Sindhi, Tharparkar, Kankrej
- **Buffalo**: Murrah, Nili-Ravi, Jaffrabadi, Mehsana, Pandharpuri

Add more breeds by updating `BREED_DATABASE` in `app.py`.

## API Endpoints

### POST `/predict`
Upload an image and get breed prediction.

**Request**:
- `image`: Image file (JPG, PNG, WEBP, max 10MB)

**Response**:
```json
{
  "breed": "Gir",
  "confidence": 0.85,
  "top_predictions": [
    {"breed": "Gir", "confidence": 0.85},
    {"breed": "Sahiwal", "confidence": 0.10},
    {"breed": "Kankrej", "confidence": 0.05}
  ],
  "explanation": "Long curved horns detected...",
  "breed_info": {
    "category": "Cattle",
    "milk_yield": "2000-3000 liters/year",
    "region": "Gujarat"
  }
}
```

### GET `/health`
Check API health status.

## Project Structure

```
.
├── app.py              # Flask backend API
├── index.html          # Frontend web interface
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── uploads/           # Uploaded images (created automatically)
```

## Development Notes

- The current implementation uses a demo prediction function
- Replace `predict_breed_demo()` with your trained model for production
- Adjust `preprocess_image()` based on your model's input requirements
- Update `BREED_DATABASE` with complete breed information

## Future Enhancements

- [ ] Multi-view image support (face + side + horns)
- [ ] Offline mode with TensorFlow.js
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Integration with government databases

## License

Built for Smart India Hackathon - Agriculture / Animal Husbandry / AI

## Support

For issues or questions, please refer to the project documentation or contact the development team.
