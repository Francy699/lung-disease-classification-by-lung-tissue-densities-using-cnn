🫁 Lung Disease Classification using Deep Learning
This project is a deep learning-based web application for classifying lung diseases from chest X-ray images using the DenseNet121 model. The user can upload a chest X-ray image, and the system predicts the type of lung disease using a trained convolutional neural network (CNN).

📌 Features
Upload chest X-ray images (JPG/PNG)

Predict lung disease with high accuracy

User-friendly Streamlit web interface

Uses pre-trained DenseNet121 model fine-tuned for classification

Real-time prediction with visualization

🧠 Model Architecture
Backbone: DenseNet121 (pretrained on ImageNet)

Framework: TensorFlow/Keras

Classifier Head: Fully connected layers for multiclass classification

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

🩺 Lung Disease Classes
The model is trained to classify the following lung diseases:

Normal

COVID-19

Pneumonia

Tuberculosis

Other (customizable depending on dataset)

📁 Project Structure
bash
Copy
Edit
lung-disease-classification/
│
├── model/
│   └── densenet_model.h5               # Trained model
│
├── app.py                              # Streamlit app code
├── lung_utils.py                       # Helper functions (preprocessing, prediction, etc.)
├── requirements.txt                    # List of dependencies
├── README.md                           # Project documentation
│
├── data/
│   └── sample_images/                  # Example test images
│
├── assets/
│   └── logo.png                        # Project logo/image
🚀 How to Run the App
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/lung-disease-classification.git
cd lung-disease-classification
Create virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate    # On Windows use venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
🧪 Dataset
This project uses chest X-ray images from publicly available datasets such as:

COVID-19 Radiography Database

NIH Chest X-ray Dataset

Make sure to place your training and test data in appropriate folders if you're retraining the model.

📊 Evaluation Metrics

Metric	Value (Example)
Accuracy	94.7%
Precision	93.8%
Recall	94.1%
F1 Score	93.9%
You can include your actual evaluation metrics here.

📷 Sample Prediction

🤝 Contributors
Francy - GitHub

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

