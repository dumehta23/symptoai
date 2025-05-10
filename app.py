from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from datetime import datetime
import pickle
import pandas as pd
import os
import numpy as np
import traceback

 
CONFIG = {
    'SECRET_KEY': os.urandom(24),
    'SQLALCHEMY_DATABASE_URI': 'mssql+pyodbc://drashtimehta:Drashti23@addpatientserver.database.windows.net/patientDatabase?driver=ODBC+Driver+18+for+SQL+Server',
    'SQLALCHEMY_ENGINE_OPTIONS': {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_timeout': 30,
        'pool_size': 10,
        'max_overflow': 20,
    },
    'SQLALCHEMY_TRACK_MODIFICATIONS': False
}

API_SECRET_KEY = "CPSC597"
DEFAULT_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 
    'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'cough', 
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'back_pain', 'constipation', 'abdominal_pain', 
    'diarrhea', 'mild_fever', 'yellowing_of_eyes', 'swelling_of_stomach',
    'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm',
    'throat_irritation', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate'
]

 
app = Flask(__name__)
app.config.update(CONFIG)

 
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

 
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    blood_type = db.Column(db.String(3), nullable=False)  
    medical_condition = db.Column(db.String(200), nullable=False)
    medication = db.Column(db.String(200), nullable=False)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

 
def load_symptoms():
    try:
        with open('models/symptom_features.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return DEFAULT_SYMPTOMS

def organize_symptoms(symptoms):
    categories = {
        'Respiratory': ['cough', 'breathlessness', 'sneezing', 'phlegm', 'throat_irritation'],
        'Digestive': ['stomach_pain', 'vomiting', 'nausea', 'diarrhea', 'constipation', 'indigestion'],
        'Neurological': ['headache', 'dizziness', 'fatigue', 'lethargy', 'mood_swings'],
        'Skin': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'yellowish_skin'],
        'Pain': ['joint_pain', 'back_pain', 'chest_pain', 'abdominal_pain'],
        'General': ['fever', 'chills', 'sweating', 'weight_loss', 'weight_gain']
    }
    
    all_categorized = sum(categories.values(), [])
    uncategorized = [s for s in symptoms if s not in all_categorized]
    categories['General'].extend(uncategorized)
    
    return {cat: [sublist] for cat, sublist in categories.items()}

def make_prediction(selected_symptoms):
    model = load_model('models/cnn_disease_model.h5')
    with open('models/cnn_label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    
    symptom_features = load_symptoms()
    input_data = np.zeros(len(symptom_features))
    
    for symptom in selected_symptoms:
        if symptom in symptom_features:
            input_data[symptom_features.index(symptom)] = 1
    
    input_data = input_data.reshape(1, len(input_data), 1)
    prediction_proba = model.predict(input_data)[0]
    predicted_class_idx = np.argmax(prediction_proba)
    prediction = le.inverse_transform([predicted_class_idx])[0]
    confidence = prediction_proba[predicted_class_idx] * 100
    
    top5_indices = np.argsort(prediction_proba)[-5:][::-1]
    top_diseases = [
        (le.inverse_transform([idx])[0], prediction_proba[idx]*100)
        for idx in top5_indices
    ]
    
    return prediction, confidence, top_diseases

def create_admin():
    if not User.query.filter_by(username='admin').first():
        admin = User(
            username='admin',
            password=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()

 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            selected_symptoms = request.form.getlist('symptoms')
            prediction, confidence, top_diseases = make_prediction(selected_symptoms)
            
            return render_template('results.html', 
                                prediction=prediction,
                                confidence=f"{confidence:.2f}%",
                                top_diseases=top_diseases,
                                selected_symptoms=selected_symptoms)
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "error")
            print(traceback.format_exc())
    
    symptoms = load_symptoms()
    symptom_categories = organize_symptoms(symptoms)
    return render_template('predict.html',
                         symptom_categories=symptom_categories,
                         all_symptoms=symptoms)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('view_patients'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if request.method == 'POST':
        try:
            patient = Patient(
                name=request.form['name'],
                age=int(request.form['age']),
                gender=request.form['gender'],
                blood_type=request.form['blood_type'],
                medical_condition=request.form['medical_condition'],
                medication=request.form['medication']
            )
            db.session.add(patient)
            db.session.commit()
            flash('Patient added successfully!', 'success')
            return redirect(url_for('view_patients'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding patient: {str(e)}', 'error')
    return render_template('add_patient.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if request.headers.get('x-api-key') != API_SECRET_KEY:
            return jsonify({'error': 'Unauthorized access. Invalid API Key.'}), 401

        selected_symptoms = request.get_json().get('symptoms', [])
        prediction, confidence, _ = make_prediction(selected_symptoms)
        
        return jsonify({
            'prediction': prediction.replace('_', ' ').title(),
            'confidence': f"{confidence:.2f}%",
            'symptoms_checked': selected_symptoms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/patients')
@login_required
def view_patients():
    patients = Patient.query.all()
    return render_template('patients.html', patients=patients)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        try:
            feedback = Feedback(
                name=request.form['name'],
                email=request.form['email'],
                message=request.form['message'],
                rating=int(request.form['rating'])
            )
            db.session.add(feedback)
            db.session.commit()
            flash('Thank you for your feedback!', 'success')
            return redirect(url_for('feedback'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error submitting feedback: {str(e)}', 'error')
    return render_template('feedback.html')


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin()
    app.run(debug=True)
