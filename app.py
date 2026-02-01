from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import io
import json
import re
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Breed database (Indian cattle and buffalo breeds)
BREED_DATABASE = {
    'Gir': {
        'category': 'Cattle',
        'milk_yield': '2000-3000 liters/year',
        'region': 'Gujarat',
        'traits': 'Long curved horns, prominent hump, red-white coat'
    },
    'Sahiwal': {
        'category': 'Cattle',
        'milk_yield': '2000-3000 liters/year',
        'region': 'Punjab, Haryana',
        'traits': 'Reddish brown coat, medium-sized hump'
    },
    'Red Sindhi': {
        'category': 'Cattle',
        'milk_yield': '1800-2500 liters/year',
        'region': 'Sindh region',
        'traits': 'Deep red color, medium-sized body'
    },
    'Tharparkar': {
        'category': 'Cattle',
        'milk_yield': '2000-2800 liters/year',
        'region': 'Rajasthan',
        'traits': 'White/grey coat, medium hump'
    },
    'Kankrej': {
        'category': 'Cattle',
        'milk_yield': '1500-2500 liters/year',
        'region': 'Gujarat, Rajasthan',
        'traits': 'Grey coat, large hump, long horns'
    },
    'Murrah': {
        'category': 'Buffalo',
        'milk_yield': '1500-2500 liters/year',
        'region': 'Haryana, Punjab',
        'traits': 'Black coat, tightly curled horns'
    },
    'Nili-Ravi': {
        'category': 'Buffalo',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Punjab',
        'traits': 'Black coat, white markings on face'
    },
    'Jaffrabadi': {
        'category': 'Buffalo',
        'milk_yield': '2000-3000 liters/year',
        'region': 'Gujarat',
        'traits': 'Large body, black coat, curved horns'
    },
    'Mehsana': {
        'category': 'Buffalo',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Gujarat',
        'traits': 'Black coat, medium-sized body'
    },
    'Pandharpuri': {
        'category': 'Buffalo',
        'milk_yield': '1200-1800 liters/year',
        'region': 'Maharashtra',
        'traits': 'Long curved horns, black coat'
    },
    'Jersey': {
        'category': 'Cattle',
        'milk_yield': '4000-6000 liters/year',
        'region': 'Jersey Island (Imported to India)',
        'traits': 'Small to medium size, fawn color, docile temperament'
    },
    'Vechur': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Kerala',
        'traits': 'Smallest cattle breed, light brown to grey coat, short horns'
    },
    'Amritmahal': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Karnataka',
        'traits': 'Grey-white coat, long horns, strong and hardy'
    },
    'Banni': {
        'category': 'Buffalo',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Gujarat',
        'traits': 'Black coat, medium-sized body, good milk yield'
    },
    'Bargur': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Tamil Nadu',
        'traits': 'Brown to dark brown coat, medium-sized, hardy'
    },
    'Dangi': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Maharashtra',
        'traits': 'White or grey coat, medium-sized, good draught animal'
    },
    'Deoni': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Karnataka, Maharashtra',
        'traits': 'White with black patches, medium-sized, dual-purpose'
    },
    'Hallikar': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Karnataka',
        'traits': 'Grey-white coat, long horns, excellent draught breed'
    },
    'Hariana': {
        'category': 'Cattle',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Haryana, Punjab',
        'traits': 'White to light grey coat, medium-sized, dual-purpose'
    },
    'Kangayam': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Tamil Nadu',
        'traits': 'Grey-white coat, strong build, excellent draught breed'
    },
    'Kasargod': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Kerala',
        'traits': 'Grey to brown coat, medium-sized, hardy'
    },
    'Kenkatha': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Uttar Pradesh',
        'traits': 'Grey-white coat, small to medium size, good draught animal'
    },
    'Kherigarh': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Uttar Pradesh',
        'traits': 'Grey-white coat, medium-sized, draught breed'
    },
    'Khillari': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Maharashtra, Karnataka',
        'traits': 'Grey-white coat, long horns, excellent draught breed'
    },
    'Malnad_gidda': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Karnataka',
        'traits': 'Small size, various colors, hardy and adaptable'
    },
    'Nagori': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Rajasthan',
        'traits': 'White to light grey coat, medium-sized, draught breed'
    },
    'Nagpuri': {
        'category': 'Buffalo',
        'milk_yield': '1200-1800 liters/year',
        'region': 'Maharashtra',
        'traits': 'Black coat, long curved horns, medium-sized'
    },
    'Nili_Ravi': {
        'category': 'Buffalo',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Punjab',
        'traits': 'Black coat, white markings on face and legs'
    },
    'Nimari': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Madhya Pradesh',
        'traits': 'Reddish brown coat, medium-sized, dual-purpose'
    },
    'Ongole': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Andhra Pradesh',
        'traits': 'White to light grey coat, large hump, strong build'
    },
    'Pulikulam': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Tamil Nadu',
        'traits': 'Grey-white coat, small to medium size, hardy'
    },
    'Rathi': {
        'category': 'Cattle',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Rajasthan',
        'traits': 'Brown to dark brown coat, medium-sized, good milk yield'
    },
    'Surti': {
        'category': 'Buffalo',
        'milk_yield': '1500-2000 liters/year',
        'region': 'Gujarat',
        'traits': 'Black coat, medium-sized, good milk yield'
    },
    'Umblachery': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Tamil Nadu',
        'traits': 'Grey-white coat, small size, hardy breed'
    },
    'Red_Sindhi': {
        'category': 'Cattle',
        'milk_yield': '1800-2500 liters/year',
        'region': 'Sindh region',
        'traits': 'Deep red color, medium-sized body'
    },
    # Common Cross-breeds (calculated mathematically)
    'Holstein_Friesian': {
        'category': 'Cattle',
        'milk_yield': '5000-7000 liters/year',
        'region': 'Imported to India',
        'traits': 'Large black and white patches, high milk production'
    },
    'Brown_Swiss': {
        'category': 'Cattle',
        'milk_yield': '4000-6000 liters/year',
        'region': 'Imported to India',
        'traits': 'Brown to grey-brown coat, large size, dual-purpose'
    },
    'Ayrshire': {
        'category': 'Cattle',
        'milk_yield': '4000-5500 liters/year',
        'region': 'Imported to India',
        'traits': 'Red and white patches, medium to large size'
    },
    'Guernsey': {
        'category': 'Cattle',
        'milk_yield': '3500-5000 liters/year',
        'region': 'Imported to India',
        'traits': 'Fawn and white color, medium size, high butterfat milk'
    },
    'Bhadawari': {
        'category': 'Buffalo',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Uttar Pradesh, Madhya Pradesh',
        'traits': 'Black coat, medium-sized, good for ghee production'
    },
    'Chhattisgarhi': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Chhattisgarh',
        'traits': 'Grey-white coat, medium-sized, hardy breed'
    },
    'Krishna_Valley': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Karnataka, Maharashtra',
        'traits': 'Grey-white coat, large size, dual-purpose'
    },
    'Red_Dane': {
        'category': 'Cattle',
        'milk_yield': '3500-5000 liters/year',
        'region': 'Imported to India',
        'traits': 'Red coat, large size, high milk production'
    },
    'Toda': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Tamil Nadu',
        'traits': 'White coat, small to medium size, hardy'
    },
    'chilika': {
        'category': 'Buffalo',
        'milk_yield': '1200-1800 liters/year',
        'region': 'Odisha',
        'traits': 'Black coat, medium-sized, adapted to coastal regions'
    },
    'gojri': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Rajasthan',
        'traits': 'Grey-white coat, medium-sized, hardy'
    },
    'kalahandi': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Odisha',
        'traits': 'Grey-white coat, small to medium size, hardy'
    },
    'luit': {
        'category': 'Cattle',
        'milk_yield': '800-1200 liters/year',
        'region': 'Assam',
        'traits': 'Grey-white coat, medium-sized, adapted to humid climate'
    },
    'marathwada': {
        'category': 'Cattle',
        'milk_yield': '1000-1500 liters/year',
        'region': 'Maharashtra',
        'traits': 'Grey-white coat, medium-sized, dual-purpose'
    },
    'pandharpuri': {
        'category': 'Buffalo',
        'milk_yield': '1200-1800 liters/year',
        'region': 'Maharashtra',
        'traits': 'Long curved horns, black coat'
    }
}

# Cache for cross-breed calculations to avoid recomputation
_crossbreed_cache = {}

def calculate_crossbreed_info(breed_name):
    """
    Calculate cross-breed information mathematically based on parent breeds.
    Handles common cross-breed naming patterns. Optimized with caching.
    """
    # Check cache first
    if breed_name in _crossbreed_cache:
        return _crossbreed_cache[breed_name]
    
    breed_lower = breed_name.lower().replace('_', ' ').replace('-', ' ').strip()
    
    # Quick pattern matching - most common cases first
    parent_breeds = []
    
    # Fast pattern checks (most common first)
    if 'jersey' in breed_lower:
        parent_breeds = ['Jersey']
    elif 'holstein' in breed_lower or 'hf' in breed_lower:
        parent_breeds = ['Holstein_Friesian']
    elif 'gir' in breed_lower and 'x' in breed_lower:
        parent_breeds = ['Gir']
    elif 'sahiwal' in breed_lower and 'x' in breed_lower:
        parent_breeds = ['Sahiwal']
    elif 'murrah' in breed_lower and 'x' in breed_lower:
        parent_breeds = ['Murrah']
    else:
        # Quick check against database keys (limited to first 20 matches for speed)
        breed_lower_clean = breed_lower.replace(' ', '').replace('-', '').replace('_', '')
        for known_breed in list(BREED_DATABASE.keys())[:20]:  # Limit search for speed
            known_lower = known_breed.lower().replace('_', '').replace('-', '')
            if len(known_lower) > 4 and known_lower in breed_lower_clean:
                parent_breeds = [known_breed]
                break
    
    # Calculate and cache result
    if parent_breeds:
        result = calculate_from_parents(parent_breeds, breed_name)
    else:
        result = {
            'category': 'Cross-breed',
            'milk_yield': '1500-2500 liters/year (estimated)',
            'region': 'Various regions',
            'traits': 'Mixed characteristics from parent breeds'
        }
    
    _crossbreed_cache[breed_name] = result
    return result

def calculate_from_parents(parent_breeds, breed_name):
    """
    Calculate cross-breed characteristics from parent breed data.
    Uses weighted averages and combinations. Optimized for speed.
    """
    if not parent_breeds:
        return None
    
    # Get parent breed data (optimized - only get first parent for speed)
    parent = parent_breeds[0]
    parent_info = BREED_DATABASE.get(parent, {})
    
    if not parent_info:
        return None
    
    # Quick milk yield calculation (simplified)
    milk_yield = '1500-2500 liters/year (estimated)'
    if 'milk_yield' in parent_info:
        # Extract numbers quickly
        numbers = re.findall(r'\d+', parent_info['milk_yield'])
        if len(numbers) >= 2:
            avg = (int(numbers[0]) + int(numbers[1])) / 2
            # Apply 85% factor for cross-breed
            min_yield = int(avg * 0.75)
            max_yield = int(avg * 0.95)
            milk_yield = f'{min_yield}-{max_yield} liters/year (estimated)'
    
    # Use parent's category and region (simplified)
    category = parent_info.get('category', 'Cross-breed')
    region = parent_info.get('region', 'Various regions')
    
    # Simplified traits
    traits = parent_info.get('traits', 'Mixed characteristics from parent breeds')
    if ',' in traits:
        traits = f'Mixed characteristics: {traits.split(",")[0]}'
    
    return {
        'category': category,
        'milk_yield': milk_yield,
        'region': region,
        'traits': traits
    }

# Explanation templates for different breeds
EXPLANATION_TEMPLATES = {
    'Gir': 'Long curved horns detected, prominent hump observed, red-white coat pattern matched. These are characteristic features of Gir cattle.',
    'Sahiwal': 'Reddish brown coat identified, medium-sized hump detected. Typical Sahiwal breed characteristics.',
    'Red Sindhi': 'Deep red coloration observed, medium body size. Consistent with Red Sindhi breed traits.',
    'Tharparkar': 'White/grey coat pattern detected, medium hump. Typical Tharparkar breed features.',
    'Kankrej': 'Grey coat identified, large hump and long horns detected. Characteristic of Kankrej breed.',
    'Murrah': 'Black coat with tightly curled horns observed. Classic Murrah buffalo traits.',
    'Nili-Ravi': 'Black coat with white facial markings detected. Distinctive Nili-Ravi buffalo features.',
    'Jaffrabadi': 'Large body size, black coat, and curved horns identified. Typical Jaffrabadi buffalo characteristics.',
    'Mehsana': 'Black coat with medium body size detected. Consistent with Mehsana buffalo breed.',
    'Pandharpuri': 'Long curved horns and black coat observed. Characteristic Pandharpuri buffalo features.'
}

MODEL = None
INV_CLASS_INDICES = None
GATE_MODEL = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upscale_image(image, target_size=224):
    """
    Upscale image using high-quality resampling if it's too small.
    """
    width, height = image.size
    if width < target_size or height < target_size:
        # Calculate scale factor to maintain aspect ratio
        scale = max(target_size / width, target_size / height) * 1.1  # 10% extra for safety
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Use LANCZOS resampling for high-quality upscaling
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def validate_image_quality(image):
    """
    Validate image quality: brightness only. Auto-upscale low resolution images.
    Returns (is_valid, error_message, upscaled_image)
    """
    original_image = image.copy()
    width, height = image.size
    
    # Auto-upscale if resolution is too low instead of rejecting
    min_resolution = 224
    if width < min_resolution or height < min_resolution:
        image = upscale_image(image, min_resolution)
        # Continue processing with upscaled image
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    # Check brightness (average pixel intensity)
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    avg_brightness = np.mean(gray)
    
    # More lenient brightness checks - only reject extreme cases
    if avg_brightness < 15:  # Very dark (reduced from 30)
        return False, "Image is too dark. Please upload an image with better lighting.", image
    if avg_brightness > 240:  # Very bright (increased from 225)
        return False, "Image is too bright (overexposed). Please upload an image with better lighting.", image
    
    # Removed blur check - too aggressive, let the model handle it
    
    return True, None, image

def validate_cattle_buffalo(image):
    """
    Very lenient validation - let the model decide.
    Only reject obviously wrong images based on extreme characteristics.
    Returns (is_valid, error_message)
    """
    # Remove strict validation - let the model's confidence score determine validity
    # This prevents false rejections of valid cattle/buffalo images
    # We'll rely on the model's prediction confidence instead
    
    # Only check if image is completely uniform (likely not an animal photo)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    color_variance = np.var(img_array, axis=(0, 1))
    avg_variance = np.mean(color_variance)
    
    # Only reject if variance is extremely low (likely a solid color background)
    # Much more lenient threshold
    if avg_variance < 50:  # Reduced from 500 - only reject truly uniform images
        return False, "Image doesn't appear to contain a cattle or buffalo. Please upload an image of a cow or buffalo."
    
    # Removed human skin tone detection - too many false positives
    # The model will handle classification, and low confidence will be caught later
    
    return True, None

def preprocess_image(image):
    image = image.resize((224, 224))
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # 🔥 CRITICAL LINE

    return img_array


def _lazy_load_gate_model():
    """
    Load a lightweight ImageNet model for subject gating (human vs bovine-ish).
    This prevents obvious non-cattle images (like humans) from being classified as a breed.
    """
    global GATE_MODEL
    if GATE_MODEL is not None:
        return
    try:
        from tensorflow import keras  # type: ignore
        GATE_MODEL = keras.applications.MobileNetV2(weights="imagenet")
        print("Loaded ImageNet gate model (MobileNetV2).")
    except Exception as exc:
        print(f"Could not load gate model. Subject gating disabled. Reason: {exc}")
        GATE_MODEL = None


def gate_reject_non_bovine(image: Image.Image):
    """
    Returns (is_ok, error_message).
    Uses ImageNet predictions to reject obvious non-bovine images (especially humans).
    """
    _lazy_load_gate_model()
    if GATE_MODEL is None:
        # If gate model unavailable, don't block.
        return True, None

    try:
        from tensorflow import keras  # type: ignore
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions  # type: ignore

        img = image.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        preds = GATE_MODEL.predict(arr, verbose=0)
        top = decode_predictions(preds, top=5)[0]
        # top entries: (class_id, label, prob)

        labels = [(label.lower(), float(prob)) for _, label, prob in top]

        bovine_keywords = [
            "ox",
            "water_buffalo",
            "buffalo",
            "cow",
            "bull",
            "bison",
            "cattle",
            "zebu",
        ]

        # Non-bovine animals that should be rejected
        non_bovine_animals = [
            "dog",
            "puppy",
            "cat",
            "kitten",
            "horse",
            "pony",
            "pig",
            "sheep",
            "goat",
            "chicken",
            "rooster",
            "duck",
            "bird",
            "eagle",
            "lion",
            "tiger",
            "elephant",
            "bear",
            "wolf",
            "fox",
            "deer",
            "rabbit",
            "hamster",
            "mouse",
            "rat",
            "squirrel",
            "monkey",
            "ape",
            "chimpanzee",
            "gorilla",
            "zebra",
            "giraffe",
            "camel",
            "llama",
            "alpaca",
        ]

        # Labels commonly triggered by text/doc/screens; treat as "not an animal photo"
        texty_keywords = [
            "book_jacket",
            "menu",
            "comic_book",
            "newspaper",
            "web_site",
            "website",
            "screen",
            "monitor",
            "digital_clock",
            "scoreboard",
            "crossword",
            "envelope",
            "letter",
            "notebook",
        ]

        def is_bovine(label: str) -> bool:
            return any(k in label for k in bovine_keywords)

        def is_non_bovine_animal(label: str) -> bool:
            return any(animal in label for animal in non_bovine_animals)

        bovine_score = max((p for l, p in labels if is_bovine(l)), default=0.0)
        person_score = max((p for l, p in labels if l == "person"), default=0.0)
        text_score = max((p for l, p in labels if any(t in l for t in texty_keywords)), default=0.0)
        non_bovine_score = max((p for l, p in labels if is_non_bovine_animal(l)), default=0.0)

        # Reject text/doc/screens aggressively.
        if text_score >= 0.10 and bovine_score < 0.05:
            return False, "Text/document image detected. Please upload a clear cattle/buffalo photo."

        # Reject if model strongly thinks it's a person and sees no bovine evidence.
        if person_score >= 0.12 and bovine_score < 0.05:
            return False, "Human detected. Please upload a clear image of a cattle or buffalo."

        # Reject non-bovine animals (dogs, cats, horses, etc.) - more aggressive
        # Check all top labels, not just the highest
        for label, prob in labels:
            if is_non_bovine_animal(label) and prob >= 0.10:
                animal_type = label.replace('_', ' ').split(',')[0]  # Get first part of label
                return False, f"This appears to be a {animal_type}. Please upload an image of a cattle or buffalo."

        # Reject if non-bovine animal score is high and bovine score is very low
        if non_bovine_score >= 0.05 and bovine_score < 0.08:
            # Find the specific animal type
            for label, prob in labels:
                if is_non_bovine_animal(label) and prob >= 0.05:
                    animal_type = label.replace('_', ' ').split(',')[0]
                    return False, f"This appears to be a {animal_type}. Please upload an image of a cattle or buffalo."

        # Also reject if there is essentially no bovine evidence at all.
        # This still allows non-bovine cattle photos if the gate model is unsure (common),
        # but blocks clearly unrelated images.
        # If neither bovine nor person, but still clearly not bovine, reject.
        # (This catches many random objects. It may still pass some edge cases.)
        if bovine_score < 0.01 and (person_score >= 0.08 or text_score >= 0.08 or non_bovine_score >= 0.05):
            return False, "Image appears unrelated. Please upload a cattle/buffalo image."

        return True, None
    except Exception:
        # If anything goes wrong, don't block inference.
        return True, None

def predict_breed_demo(image):
    """
    Demo prediction function - replace this with your actual model
    This simulates breed detection based on image analysis
    """
    # Convert image to numpy array for analysis
    img_array = np.array(image)
    
    # Simple heuristics for demo (replace with actual model)
    # In production, load your trained model here
    # model = load_model('models/breed_classifier.h5')
    # predictions = model.predict(preprocessed_image)
    
    # Demo: Analyze image characteristics
    avg_color = np.mean(img_array, axis=(0, 1))
    brightness = np.mean(img_array)
    
    # Mock predictions based on simple heuristics
    breeds = list(BREED_DATABASE.keys())
    
    # Simulate prediction probabilities
    # In real implementation, use model.predict()
    np.random.seed(hash(str(avg_color)) % 2**32)
    probs = np.random.dirichlet(np.ones(len(breeds)) * 2)
    probs = probs / probs.sum()
    
    # Sort by probability
    sorted_indices = np.argsort(probs)[::-1]
    
    predictions = []
    for idx in sorted_indices[:5]:  # Top 5 predictions
        predictions.append({
            'breed': breeds[idx],
            'confidence': float(probs[idx])
        })
    
    return predictions


def init_model():
    """
    Try to load a real trained model if available.
    Falls back to demo predictions when model or dependencies are missing.
    """
    global MODEL, INV_CLASS_INDICES
    models_dir = os.path.join(os.getcwd(), "models")
    model_path = os.path.join(models_dir, "breed_classifier.h5")
    indices_path = os.path.join(models_dir, "class_indices.json")

    try:
        # Lazy import so that TensorFlow is optional
        from tensorflow import keras  # type: ignore

        if os.path.exists(model_path) and os.path.exists(indices_path):
            print(f"Loading breed classifier model from {model_path} ...")
            MODEL = keras.models.load_model(model_path)
            with open(indices_path, "r", encoding="utf-8") as f:
                class_indices = json.load(f)
            # Invert mapping: index -> class_name
            INV_CLASS_INDICES = {v: k for k, v in class_indices.items()}
            print("Model and class indices loaded successfully.")
        else:
            print("No trained model found in 'models/'. Using demo predictions.")
    except Exception as exc:
        print(f"Could not load TensorFlow model. Using demo predictions. Reason: {exc}")


def predict_breed(image):
    if MODEL is None or INV_CLASS_INDICES is None:
        print("⚠️ USING DEMO MODEL")
        return predict_breed_demo(image)

    print("✅ USING REAL TRAINED MODEL")

    preprocessed = preprocess_image(image)
    probs = MODEL.predict(preprocessed)[0]

    print("Raw probs:", probs)
    print("Max prob:", float(np.max(probs)))

    breeds = [INV_CLASS_INDICES[i] for i in range(len(probs))]
    probs = probs / probs.sum()

    sorted_indices = np.argsort(probs)[::-1]
    predictions = []
    for idx in sorted_indices[:5]:
        predictions.append(
            {
                "breed": breeds[idx],
                "confidence": float(probs[idx]),
            }
        )
    return predictions


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is present
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload JPG, PNG, or WEBP"}), 400
        
        # Read and process image
        image_bytes = file.read()
        if len(image_bytes) > MAX_FILE_SIZE:
            return jsonify({"error": "File size exceeds 10MB limit"}), 400
        
        image = Image.open(io.BytesIO(image_bytes))
        
        # Validate image quality (auto-upscales if needed)
        is_valid, quality_error, processed_image = validate_image_quality(image)
        if not is_valid:
            return jsonify({
                "error": quality_error,
                "error_type": "quality"
            }), 400
        
        # Use processed image (may be upscaled)
        image = processed_image
        
        # Very lenient subject validation - mostly rely on model confidence
        is_valid, animal_error = validate_cattle_buffalo(image)
        if not is_valid:
            return jsonify({
                "error": animal_error,
                "error_type": "subject"
            }), 400

        # Stronger subject gate: reject obvious humans/unrelated subjects before breed prediction
        # This MUST run before breed prediction to catch dogs and other animals
        """ ok, gate_error = gate_reject_non_bovine(image)
        if not ok:
            return jsonify({
                "error": gate_error,
                "error_type": "subject"
            }), 400 """

        # Preprocess image
        # Get predictions (real model if available, otherwise demo)
        predictions = predict_breed(image)
        
        # Additional post-prediction gate: Check if predicted breed makes sense
        # If model predicts a breed but gate model strongly suggests non-bovine, reject
        if predictions:
            top_prediction = predictions[0]
            breed_name = top_prediction['breed']
            # Double-check with gate model if confidence is suspiciously low
            if top_prediction['confidence'] < 0.30:
                # Re-run gate check to be extra sure
                ok, gate_error = gate_reject_non_bovine(image)
                if not ok:
                    return jsonify({
                        "error": gate_error,
                        "error_type": "subject"
                    }), 400
        
        if not predictions:
            return jsonify({"error": "Could not detect breed"}), 500
        
        # Get top prediction
        top_prediction = predictions[0]
        breed_name = top_prediction['breed']
        confidence = top_prediction['confidence']

        # If model predicts "Other", treat as invalid subject (human/text/random)
        if str(breed_name).strip().lower() == "other":
            return jsonify({
                "error": "This image does not look like a cattle/buffalo breed photo (classified as Other). Please upload a clear cattle/buffalo image.",
                "error_type": "subject"
            }), 400
        
        # Reject if confidence is too low (often means wrong subject or unclear image)
        # Lower threshold to allow more valid images through
        if confidence < 0.05:
            return jsonify({
                "error": "Unable to identify breed with sufficient confidence. Please upload a clearer, closer cattle/buffalo image (preferably side/face view).",
                "error_type": "subject"
            }), 400
        
        # Get breed information (handle name variations)
        # Try multiple name formats
        breed_info = BREED_DATABASE.get(breed_name, {})
        if not breed_info:
            # Try with underscore replaced by hyphen
            breed_key = breed_name.replace('_', '-')
            breed_info = BREED_DATABASE.get(breed_key, {})
        if not breed_info:
            # Try with hyphen replaced by underscore
            breed_key = breed_name.replace('-', '_')
            breed_info = BREED_DATABASE.get(breed_key, {})
        if not breed_info:
            # Try with spaces replaced by underscore
            breed_key = breed_name.replace(' ', '_')
            breed_info = BREED_DATABASE.get(breed_key, {})
        
        # If breed info not found, try to calculate for cross-breeds
        if not breed_info:
            breed_info = calculate_crossbreed_info(breed_name)
        
        # Get explanation
        explanation = EXPLANATION_TEMPLATES.get(
            breed_name,
            f"Detected characteristics match {breed_name} breed features.",
        )
        
        response = {
            "breed": breed_name,
            "confidence": confidence,
            "top_predictions": predictions[:3],  # Top 3 predictions
            "breed_info": breed_info,
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "message": "Breed identification API is running"})

init_model()
if __name__ == "__main__":
    print("Starting Cattle & Buffalo Breed Identification API...")
    app.run()



