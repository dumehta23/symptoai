# AI Diagnostic System

## Project Overview
This project is an AI-powered diagnostic system designed to assist healthcare professionals by providing real-time diagnostic recommendations based on patient input. It features a user-friendly graphical interface built with Python, HTML, and CSS making it accessible even to non-technical users. The system utilizes machine learning models trained on simulated healthcare datasets to deliver accurate disease diagnosis and predictions.  

Designed for both personal use and healthcare professionals, it offers:    
- Patient diagnosis based on inputed symptoms
- Multi-model AI (CNN, RNN, Random Forest) for robust predictions
- Azure SQL Database for secure patient data storage
- API for EHR integration

Built with Python (Flask), HTML/CSS, and database on Azure.

## Features
- Real-time symptom analysis with confidence scoring
- Patient management dashboard (add/view/update records)
- API for EHR integration

## Setup Instructions

### Prerequisites
Python 3.8+
Azure account (for SQL Database)

### 1. Clone the Repository
```bash
git clone https://github.com/dumehta23/symptoai
cd symptoai
pip install -r requirements.txt
```

### 2. Install Required Packages
- Python 3.8+
- TensorFlow
- PyTorch
- NumPy
- Pandas
- Scikit-Learn
- Flask
- Flask-SQLAlchemy
- Flask-Login

### 3. Database Configuration 
1. Automatic setup: Database will be created automatically at instance/patient_feedback.db on first run.
Backups are stored in the same folder with timestamped filenames
2. Azure SQL Setup:
    Create a server (addpatientserver.database.windows.net)
    Set firewall rules to allow your IP
    Create database patientDatabase
3. Configure connection in app.py:
    app.config['SQLALCHEMY_DATABASE_URI'] = (
        'mssql+pyodbc://{username}:{password}@{server}/{database}?'
        'driver=ODBC+Driver+18+for+SQL+Server'
    ).format(
        username='your_admin',
        password='your_password',
        server='addpatientserver.database.windows.net',
        database='patientDatabase'
    )

### 4. Download the Dataset
You must manually download the dataset from Kaggle:

- [Diseases and Symptoms Dataset on Kaggle](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)

Once downloaded:
- Place the dataset (CSV file) into the project's `data/` folder.

> **Note:** Create the `data/` folder if it does not exist.

## Project Structure
```
/data                         # Dataset files
    /dataset from Kaggle
/models                       # Trained model files
    /disease_model.pkl
    /symptom_features.pkl
    /disease_model.csv
    /symptom_features.csv
    /symptom_importance.csv
    /cnn_disease_model.h5
    /cnn_label_encoder.pkl
/static
    /css
        /style.css
    /homepage.jpg
/templates
    /add_patient.html
    /base.html
    /index.html
    /login.html
    /patients.html
    /predict.html
    /results.html
README.md
app.py 
CNN_Testing.ipynb
RNN_Testing.ipynb
RandomForest_Testing.ipynb
train_model.py
requirements.txt
api-docs.md
test_api.py          
```

## How to Run
Run the main Python script to start the application:
```bash
# Terminal 1: Start Flask
python app.py

# Terminal 2: Test API (EHR data)
python test_api.py 
* This is running with simulated ehr_data.csv*
```
A GUI window of a web application will open where you can navigate SymptoAI.

## For Clinicians Only: How to Use Real EHR Data:
1. Prepare your CSV file with these columns (example below):
    patient_id,name,age,gender,medical_history,symptoms,notes
2. Save it as ehr_data.csv in the main project folder
3. Example CSV Format:
    1,John Smith,45,Male,"Diabetes, Hypertension","fever, cough","Patient reports 3-day fever"
    2,Maria Garcia,32,Female,None,"headache","Morning headaches for 1 week"

 ## Database Management
1. Access Data: Use SQLite browsers or sqlite3 instance/patient_feedback.db
2. Backups: Automatic daily backups in the instance folder
3. Azure: To switch to Azure SQL, uncomment the configuration in app.py

## Future Work
- Multilingual support.
- Broader EHR system compatibility.
- Descriptions of diagnosis. 
- Continuous model training.
- Mobile application development.

## License
This project uses open-source libraries and public datasets. 

---
