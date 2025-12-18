import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('teleconnect.csv')  # Update with full path if necessary

# Handle TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Target variable: Churn
le = LabelEncoder()
y = le.fit_transform(df['Churn'])  # Yes=1, No=0

# Features
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

features = categorical_features + numerical_features
X_raw = df[features]

# Preprocessing
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
], remainder='passthrough')

X = ct.fit_transform(X_raw)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"Dataset shape: {X_scaled.shape}")
print(f"Churn distribution: No={np.bincount(y)[0]}, Yes={np.bincount(y)[1]}")

# Function to build optimized classifier
def build_optimized_classifier(hidden_layers=[128, 64, 32], dropout_rate=0.3):
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to train and return all metrics
def train_and_evaluate(optimizer, optimizer_name, epochs=100):
    model = build_optimized_classifier()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                        validation_split=0.2, verbose=1, callbacks=[early_stop])
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    val_loss = min(history.history['val_loss'])
    
    print(f"\n{optimizer_name} Results:")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | Best Val Loss: {val_loss:.4f}")
    
    return acc, prec, rec, f1, val_loss, history, y_pred

# === ORIGINAL SIMPLE MODEL ===
print("\n=== Training Original Simple Model ===")
original_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
original_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

orig_history = original_model.fit(X_train, y_train, epochs=50, batch_size=32,
                                  validation_split=0.2, verbose=1)

orig_pred_prob = original_model.predict(X_test, verbose=0)
orig_pred = (orig_pred_prob > 0.5).astype(int).flatten()

orig_acc = accuracy_score(y_test, orig_pred)
orig_prec = precision_score(y_test, orig_pred)
orig_rec = recall_score(y_test, orig_pred)
orig_f1 = f1_score(y_test, orig_pred)
orig_loss = min(orig_history.history['val_loss'])

print(f"\nOriginal Model Results:")
print(f"Accuracy: {orig_acc:.4f} | Precision: {orig_prec:.4f} | Recall: {orig_rec:.4f} | F1-Score: {orig_f1:.4f} | Best Val Loss: {orig_loss:.4f}")

# === OPTIMIZED MODELS ===
print("\n=== Training Optimized Models ===")
acc1, prec1, rec1, f11, loss1, history1, pred1 = train_and_evaluate(Adam(learning_rate=0.001), "Adam (Optimized)")
acc2, prec2, rec2, f12, loss2, history2, pred2 = train_and_evaluate(RMSprop(learning_rate=0.001), "RMSprop")
acc3, prec3, rec3, f13, loss3, history3, pred3 = train_and_evaluate(Adagrad(learning_rate=0.01), "Adagrad")

# === COMPILE ALL METRIC RESULTS ===
results = pd.DataFrame({
    'Model/Optimizer': [
        'Original (Simple + Adam)',
        'Adam (Deeper + Dropout)',
        'RMSprop',
        'Adagrad'
    ],
    'Accuracy': [orig_acc, acc1, acc2, acc3],
    'Precision': [orig_prec, prec1, prec2, prec3],
    'Recall': [orig_rec, rec1, rec2, rec3],
    'F1-Score': [orig_f1, f11, f12, f13],
    'Best Validation Loss': [orig_loss, loss1, loss2, loss3]
})

# Round for clean display
results_rounded = results.round(4)

print("\n=== Final Metric Results Table ===")
print(results_rounded)

# === SAVE METRICS TABLE AS PNG ===
fig, ax = plt.subplots(figsize=(14, 6))  # Wider for better readability
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=results_rounded.values,
                 colLabels=results_rounded.columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.3, 3)  # Make rows taller

# Style the table
table[(0, 0)].set_facecolor('#4CAF50')
table[(0, 0)].set_text_props(weight='bold', color='white')
for i in range(1, len(results_rounded.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Performance Evaluation Metrics Before and After Optimization\n'
          '(Accuracy, Precision, Recall, F1-Score, Loss Rate)',
          fontsize=16, fontweight='bold', pad=30)

plt.savefig('performance_metrics_results_table.png', bbox_inches='tight', dpi=300)
print("\nMetrics table saved as: performance_metrics_results_table.png")
plt.close()

# === ADDITIONAL VISUALS ===
# Confusion Matrix for the best performing model (example: RMSprop)
cm = confusion_matrix(y_test, pred2)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix - RMSprop (Optimized Model)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()

# Loss curves comparison
plt.figure(figsize=(12, 6))
plt.plot(orig_history.history['val_loss'], label='Original Model Val Loss', linewidth=2)
plt.plot(history2.history['val_loss'], label='RMSprop Optimized Val Loss', linewidth=2)
plt.title('Validation Loss: Before vs After Optimization')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_comparison.png', bbox_inches='tight', dpi=300)
plt.close()

print("\nAll files saved successfully:")
print("1. performance_metrics_results_table.png   ← Main required metrics in PNG format")
print("2. confusion_matrix.png")
print("3. loss_comparison.png")