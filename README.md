# Modular Iris ML Pipeline

![Python](https://img.shields.io/badge/python-3.10%2B-blue)

A modular machine learning project that trains a **Logistic Regression model** on the Iris dataset with proper code structure, reproducibility, and unit testing.

---

## 📂 Project Structure

```
random-forest-iris/
│
├── src/                 # Core modules
│   ├── data_loader.py   # Load and split Iris dataset
│   ├── preprocessing.py # Data preprocessing (scaling)
│   ├── model.py         # Model training and persistence
│   ├── evaluate.py      # Model evaluation
│   └── utils.py         # Logger and helper functions
│
├── tests/               # Unit tests for all modules
│
├── scripts/
│   └── run_training.sh  # Training automation script
│
├── main.py              # Entry point
├── requirements.txt     # Dependencies with pinned versions
└── README.md
```

---

## 🚀 Features
- **PEP 8 compliant** and well-documented code
- Modular design for reusability and scalability
- Unit tests with `pytest`
- CI/CD ready with GitHub Actions
- Reproducible environment via `requirements.txt` and `run_training.sh`

---

## 🛠 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/modular-iris-ml.git
   cd modular-iris-ml
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶ Usage

Run the full pipeline:

```bash
python main.py
```

Or use the helper script:

```bash
chmod +x scripts/run_training.sh
./scripts/run_training.sh
```

---

## ✅ Running Tests

We use `pytest` for testing.

```bash
pytest tests/ -v
```

---

## 📊 Example Output
```
INFO: Loading data...
INFO: Preprocessing data...
INFO: Training Logistic Regression model...
INFO: Evaluating model...
INFO: Model Accuracy: 0.9333
INFO: Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        10
           1       0.92      0.90      0.91        10
           2       0.90      0.90      0.90        10
```

---

## 🧪 Continuous Integration
This repository includes a GitHub Actions workflow that:
- Installs dependencies
- Runs all tests automatically on every push or pull request

---

## 📌 Requirements
- Python 3.10+
- pip 23+

---

## 📜 License
MIT License. Feel free to use and modify this project.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
