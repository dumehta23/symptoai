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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+18+for+SQL+Server'.format(
    username='drashtimehta',   
    password='Drashti23',   
    server='addpatientserver.database.windows.net',
    database='patientDatabase'
)
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_timeout': 30,
    'pool_size': 10,
    'max_overflow': 20,
}
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

API_SECRET_KEY = "CPSC597"

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

@login_manager.user_loader
def load_user(user_id):
    user = None
    for u in User.query.all():
        if u.id == int(user_id):
            user = u
            break
    return user

def load_symptoms():
    try:
        with open('models/symptom_features.pkl', 'rb') as f:
            symptoms = pickle.load(f)
        return symptoms
    except:
 
        return [
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

 
@app.route('/')
def home():
    return render_template('index.html')

def organize_symptoms(symptoms):
    categories = {
        'Respiratory': ['cough', 'breathlessness', 'sneezing', 'phlegm', 'throat_irritation'],
        'Digestive': ['stomach_pain', 'vomiting', 'nausea', 'diarrhea', 'constipation', 'indigestion'],
        'Neurological': ['headache', 'dizziness', 'fatigue', 'lethargy', 'mood_swings'],
        'Skin': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'yellowish_skin'],
        'Pain': ['joint_pain', 'back_pain', 'chest_pain', 'abdominal_pain'],
        'General': ['fever', 'chills', 'sweating', 'weight_loss', 'weight_gain']
    }
    
    all_categorized = []
    for cat_list in categories.values():
        all_categorized += cat_list
    for symptom in symptoms:
        if symptom not in all_categorized:
            categories['General'].append(symptom)
    
    grouped_categories = {}
    for category in categories:
        grouped = []
        for i in range(0, len(categories[category]), len(categories[category])):
            grouped.append(categories[category][i:i + len(categories[category])])
        grouped_categories[category] = grouped
    return grouped_categories

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load CNN model and label encoder
            model = load_model('models/cnn_disease_model.h5')
            with open('models/cnn_label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            
            symptom_features = load_symptoms()
            input_data = np.zeros(len(symptom_features))  
            
            selected_symptoms = request.form.getlist('symptoms')
            for symptom in selected_symptoms:
                if symptom in symptom_features:
                    idx = symptom_features.index(symptom)
                    input_data[idx] = 1
            
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
        user = None
        users = User.query.all()
        for u in users:
            if u.username == username:
                user = u
                break
        
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
        api_key = request.headers.get('x-api-key')
        if api_key != API_SECRET_KEY:
            return jsonify({'error': 'Unauthorized access. Invalid API Key.'}), 401

        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        model = load_model('models/cnn_disease_model.h5')
        with open('models/cnn_label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        
        symptom_features = load_symptoms()
        input_data = np.zeros(len(symptom_features))
        for symptom in symptoms:
            if symptom in symptom_features:
                idx = symptom_features.index(symptom)
                input_data[idx] = 1
                
        input_data = input_data.reshape(1, len(input_data), 1)
        prediction_proba = model.predict(input_data)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        prediction = le.inverse_transform([predicted_class_idx])[0]
        confidence = prediction_proba[predicted_class_idx] * 100
        
        return jsonify({
            'prediction': prediction.replace('_', ' ').title(),
            'confidence': f"{confidence:.2f}%",
            'symptoms_checked': symptoms
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/patients')
@login_required
def view_patients():
    patients = Patient.query.all()
    return render_template('patients.html', patients=patients)

def create_admin():
    users = User.query.all()
    found = False
    for u in users:
        if u.username == 'admin':
            found = True
            break
    if not found:
        admin = User(username='admin', password=generate_password_hash('admin123'))
        db.session.add(admin)
        db.session.commit()

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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        create_admin()
    app.run(debug=True)
