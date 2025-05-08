import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import warnings
import time
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model


# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    
    print("\nDisease distribution (top 10):")
    print(df['diseases'].value_counts().head(10))
    print(f"\nTotal unique diseases: {df['diseases'].nunique()}")
    
    symptom_columns = []
    for col in df.columns:
        if col != 'diseases':
            symptom_columns.append(col)
    
    return df, symptom_columns

def prepare_data_for_models(df, symptom_columns):
    X = df[symptom_columns]
    y = df['diseases']
    
    if X.isnull().values.any():
        print("Found missing values. Filling with 0")
        X = X.fillna(0)

    class_counts = {}
    for label in y:
        if label not in class_counts:
            class_counts[label] = 1
        else:
            class_counts[label] += 1

    diseases_to_keep = []
    for disease, count in class_counts.items():
        if count >= 2:
            diseases_to_keep.append(disease)

    mask = []
    for label in y:
        if label in diseases_to_keep:
            mask.append(True)
        else:
            mask.append(False)

    mask = pd.Series(mask)

    if sum(mask) < len(y):
        print(f"Filtering out {len(y) - sum(mask)} samples with rare diseases")
        X = X[mask]
        y = y[mask]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y
        )
        print("Using stratified sampling for train-test split")
    except Exception as e:
        print(f"Stratified split failed: {str(e)}")
        print("Falling back to random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train_enc, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test_enc, num_classes=num_classes)

    return {
        'rf': {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X': X, 'y': y  
        },
        'feature_columns': X.columns
    }

#CNN was chosen as the model to train with because it had the highest accuracy, compared to RNN and RF.
def train_cnn(data):

    print("\nTraining CNN classifier...") 
    start_time = time.time()
    
    X_train_cnn = data['X_train'].values.reshape(data['X_train'].shape[0], 
                                               data['X_train'].shape[1], 1)
    X_test_cnn = data['X_test'].values.reshape(data['X_test'].shape[0],
                                             data['X_test'].shape[1], 1)
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(data['y_train'])
    y_test_enc = label_encoder.transform(data['y_test'])
    num_classes = len(label_encoder.classes_)
    
    y_train_cat = to_categorical(y_train_enc, num_classes=num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes=num_classes)
    
    # Build CNN model
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    # Train model
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train_cnn, y_train_cat,
                       epochs=50,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stop],
                       verbose=1)
    
    test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    training_time = time.time() - start_time
    
    print(f"\nCNN Model test accuracy: {test_acc:.4f} (trained in {training_time:.2f} seconds)")
    
    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    

    present_labels = np.unique(np.concatenate([y_test_enc, y_pred]))
    present_target_names = label_encoder.inverse_transform(present_labels)
    
    print("\nCNN Classification Report (Present Classes Only):")
    print(classification_report(y_test_enc, y_pred, 
                              labels=present_labels,
                              target_names=present_target_names))
    
 
    os.makedirs('models', exist_ok=True)
    model.save('models/cnn_disease_model.h5')
    with open('models/cnn_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/symptom_features.pkl', 'wb') as f:
        pickle.dump(list(data['X_train'].columns), f)
    
    return {
        'model': model,
        'accuracy': test_acc,
        'label_encoder': label_encoder,
        'training_time': training_time
    }


def main():
    dataset_path = 'data/Final_Augmented_dataset_Diseases_and_Symptoms.csv'
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please download the dataset from Kaggle and place it in the data directory")
        return
    
    # Check if model files already exist
    model_files = [
        'models/cnn_disease_model.h5',
        'models/cnn_label_encoder.pkl',
        'models/symptom_features.pkl'
    ]
    
    if all(os.path.exists(f) for f in model_files):
        print("\nModel files already exist. Loading pre-trained model...")
        model = load_model('models/cnn_disease_model.h5')
        with open('models/cnn_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('models/symptom_features.pkl', 'rb') as f:
            symptom_features = pickle.load(f)
        
        # Just evaluate existing model
        df, symptom_columns = load_and_preprocess_data(dataset_path)
        data = prepare_data_for_models(df, symptom_columns)
        
        X_test_cnn = data['rf']['X_test'].values.reshape(data['rf']['X_test'].shape[0],
                                                 data['rf']['X_test'].shape[1], 1)
        y_test_enc = label_encoder.transform(data['rf']['y_test'])
        y_test_cat = to_categorical(y_test_enc, num_classes=len(label_encoder.classes_))
        
        test_loss, test_acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
        print(f"\nPre-trained Model Test Accuracy: {test_acc:.4f}")
        
        y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
        present_labels = np.unique(np.concatenate([y_test_enc, y_pred]))
        present_target_names = label_encoder.inverse_transform(present_labels)
        
        print("\nClassification Report (Present Classes Only):")
        print(classification_report(y_test_enc, y_pred,
                                  labels=present_labels,
                                  target_names=present_target_names))
    else:
        print("\nNo existing model found. Training new model...")
        df, symptom_columns = load_and_preprocess_data(dataset_path)
        data = prepare_data_for_models(df, symptom_columns)
        cnn_results = train_cnn(data['rf'])
        print(f"\nCNN Results:")
        print(f"- Test Accuracy: {cnn_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()