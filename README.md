# ARTIFICIAL-NEURAL-NETWORK-MODEL-MARKETING-CAMPAIGN-PROJECT
This project implements an Artificial Neural Network (ANN) to predict customer churn using the Teleconnect dataset. The objective is to identify customers who are likely to discontinue services (churn) by learning patterns from customer demographics, service subscriptions, and billing information. Accurate churn prediction enables organizations to design proactive customer retention strategies and reduce revenue loss.

The project compares a baseline neural network model with optimized deep learning models trained using different optimizers (Adam, RMSprop, and Adagrad) to evaluate performance improvements achieved through architectural and training optimizations.

Dataset Description
Dataset name: teleconnect.csv
Target variable: Churn (Yes = 1, No = 0)
Categorical features: Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
Numerical features: SeniorCitizen, Tenure, MonthlyCharges, TotalCharges
Missing or invalid values in TotalCharges are handled by coercion and removal to ensure data integrity.

Data Preprocessing
The following preprocessing steps were applied:
Conversion of TotalCharges to numeric format and removal of missing values.
Label encoding of the target variable (Churn).
One-hot encoding of categorical features using ColumnTransformer.
Feature scaling of numerical variables using StandardScaler.
Stratified train–test split (80% training, 20% testing) to preserve class distribution.

Model Architecture
Baseline Model
Dense layers: 32 → 16 → 1
Activation functions: ReLU (hidden layers), Sigmoid (output layer)
Optimizer: Adam
Loss function: Binary Crossentropy

Optimized ANN Model
Dense layers: 128 → 64 → 32 → 1
Dropout regularization (rate = 0.3)
ReLU activations with Sigmoid output
Early stopping based on validation loss to prevent overfitting

Optimizers Evaluated
The optimized ANN architecture was trained using:
Adam (learning rate = 0.001)
RMSprop (learning rate = 0.001)
Adagrad (learning rate = 0.01)

Evaluation Metrics
Model performance was assessed using:
Accuracy
Precision
Recall
F1-score
Best validation loss
These metrics provide a balanced evaluation of predictive performance, particularly for churn detection where false negatives are costly.

Results and Visualizations
The project generates the following output files:
performance_metrics_results_table.png
Summary table comparing baseline and optimized models across all evaluation metrics.
confusion_matrix.png
Confusion matrix for the best-performing model (RMSprop), illustrating classification accuracy for churn and non-churn customers.
loss_comparison.png
Validation loss curves comparing the baseline model with the optimized RMSprop model.

Key Findings
Optimized ANN models outperform the baseline model in stability and generalization.
RMSprop demonstrated the best balance between validation loss and classification metrics.
Dropout regularization and early stopping significantly reduced overfitting.
Deep learning models are effective for capturing complex nonlinear patterns in customer churn behavior.

Technologies Used
Python 3.11
Pandas, NumPy
Scikit-learn
TensorFlow / Keras
Matplotlib, Seaborn

How to Run the Project
Ensure all dependencies are installed:
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
Place teleconnect.csv in the project directory.
Run the Python script:
python churn_prediction_ann.py
Generated results and visualizations will be saved to the working directory.

Conclusion
This project demonstrates how Artificial Neural Networks can be effectively applied to customer churn prediction. Through careful preprocessing, model optimization, and evaluation, the ANN approach provides a reliable, scalable, and data-driven solution for customer retention analytics.
