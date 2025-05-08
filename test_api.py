import requests
import csv
import random
from typing import List, Dict

# Configuration
API_URL = 'http://127.0.0.1:5000'
API_KEY = 'CPSC597'
CSV_FILE = 'data/ehr_data.csv'  

class EHRDataProcessor:
    @staticmethod
    def extract_symptoms_from_notes(notes: str) -> List[str]:
        """Extract symptoms from free-text notes using simple keyword matching"""
        symptom_keywords = {
            'fever': ['fever', 'temp', 'temperature'],
            'cough': ['cough', 'coughing'],
            'headache': ['headache', 'migraine'],
            'fatigue': ['fatigue', 'tired', 'exhaust'],
            'nausea': ['nausea', 'vomit'],
            'pain': ['pain', 'ache', 'sore'],
            'joint_pain': ['joint pain', 'arthri'],
            'shortness_of_breath': ['shortness of breath', 'sob', 'breathless'],
            'dizziness': ['dizzy', 'vertigo']
        }
        
        found_symptoms = []
        notes_lower = notes.lower()
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in notes_lower for keyword in keywords):
                found_symptoms.append(symptom)
                
        return list(set(found_symptoms))  # Remove duplicates

    @staticmethod
    def read_patients_from_csv(file_path: str) -> List[Dict]:
        """Read patient data from CSV file"""
        patients = []
        
        with open(file_path, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                structured_symptoms = []
                if 'symptoms' in row and row['symptoms']:
                    structured_symptoms = [s.strip() for s in row['symptoms'].split(',')]
                
                notes_symptoms = []
                if 'notes' in row and row['notes']:
                    notes_symptoms = EHRDataProcessor.extract_symptoms_from_notes(row['notes'])
                
                patients.append({
                    'patient_id': row.get('patient_id', ''),
                    'name': row.get('name', '').lower(),  # Lowercase for case-insensitive search
                    'age': row.get('age', ''),
                    'gender': row.get('gender', ''),
                    'medical_history': row.get('medical_history', ''),
                    'symptoms': list(set(structured_symptoms + notes_symptoms))
                })
                
        return patients

class APITester:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            'Content-Type': 'application/json',
            'x-api-key': api_key
        }
        self.patients = EHRDataProcessor.read_patients_from_csv(CSV_FILE)
    
    def find_patient(self, search_name: str) -> Dict:
        """Find patient by name (case-insensitive)"""
        search_name = search_name.lower()
        for patient in self.patients:
            if search_name in patient['name']:
                return patient
        return None
    
    def get_diagnosis(self, symptoms: List[str]) -> Dict:
        """Get diagnosis for given symptoms"""
        data = {'symptoms': symptoms}
        response = requests.post(
            f"{self.base_url}/api/predict",
            headers=self.headers,
            json=data
        )
        return response.json()
    
    def diagnose_patient(self, patient_name: str):
        """Diagnose a specific patient by name"""
        patient = self.find_patient(patient_name)
        
        if not patient:
            print(f"\nPatient '{patient_name}' not found in records")
            return
        
        print(f"\nPatient: {patient['name'].title()}")
        print(f"Age: {patient['age']}, Gender: {patient['gender']}")
        print(f"Medical History: {patient['medical_history']}")
        
        if patient['symptoms']:
            print(f"\nDetected Symptoms: {', '.join(patient['symptoms'])}")
            try:
                diagnosis = self.get_diagnosis(patient['symptoms'])
                print(f"\nDiagnosis Results:")
                print(f"Condition: {diagnosis.get('prediction', 'N/A')}")
                print(f"Confidence: {diagnosis.get('confidence', 'N/A')}")
            except Exception as e:
                print(f"Diagnosis failed: {str(e)}")
        else:
            print("No symptoms detected - cannot make diagnosis")

if __name__ == '__main__':
    tester = APITester(API_URL, API_KEY)
    
    patient_name = input("Enter patient name to diagnose: ").strip()
    tester.diagnose_patient(patient_name)
