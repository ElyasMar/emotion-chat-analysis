from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_regression_model.pkl')

# Load model
model = joblib.load(MODEL_PATH)

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def get_recommendation(emotion):
    """Get action recommendation based on emotion"""
    recommendations = {
        'anger': 'Take a deep breath and count to 10. Consider a short break.',
        'joy': 'Great! Share your positive energy with others.',
        'sadness': 'It\'s okay to feel sad. Reach out to someone you trust.',
        'fear': 'Ground yourself. Focus on what you can control.',
        'disgust': 'Step away from the situation if possible.',
        'surprise': 'Take a moment to process what happened.',
        'neutral': 'You seem calm and balanced.',
    }
    return recommendations.get(emotion, 'Stay mindful of your emotions.')

@app.route('/')
def home():
    return jsonify({
        'message': 'Emotion Detection API',
        'endpoints': {
            '/predict': 'POST - Predict emotion from text',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Predict
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        
        # Get top 3 emotions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_emotions = [
            {
                'emotion': EMOTION_LABELS[idx],
                'confidence': float(probabilities[idx])
            }
            for idx in top_3_idx
        ]
        
        primary_emotion = EMOTION_LABELS[prediction]
        recommendation = get_recommendation(primary_emotion)
        
        return jsonify({
            'text': text,
            'primary_emotion': primary_emotion,
            'confidence': float(probabilities[prediction]),
            'top_emotions': top_emotions,
            'recommendation': recommendation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)