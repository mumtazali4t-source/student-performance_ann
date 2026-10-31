# Student Performance Prediction using an Artificial Neural Network

A simple machine learning project that predicts whether a student will **Pass** or **Fail** using an Artificial Neural Network (ANN).

## Dataset

The dataset has **500 students** and **6 columns**.

| Column | Meaning |
|--------|---------|
| Study_Hours | Hours the student studies |
| Attendance | Attendance percentage |
| Previous_Marks | Marks in the previous exam |
| Assignment_Score | Score in assignments |
| Sleep_Hours | Hours of sleep |
| Result | Pass or Fail (target) |

- 250 Pass and 250 Fail records (balanced classes)
- No missing values

## What I did

1. Loaded and explored the data (shape, data types, missing values, summary statistics)
2. Plotted the number of Pass and Fail records
3. Split the data: 80% training (400 students) and 20% testing (100 students), with stratification
4. Scaled the features using `StandardScaler`
5. Trained an ANN using scikit-learn's `MLPClassifier`
6. Evaluated the model with accuracy, a classification report and a confusion matrix
7. Predicted the result for a new student

## Model

| Setting | Value |
|---------|-------|
| Hidden layers | 2 (16 and 8 neurons) |
| Activation | ReLU |
| Optimizer | Adam |
| Max iterations | 1500 |
| Early stopping | Yes (15% validation data) |
| Random state | 42 |

Training stopped early at **82 iterations**.

## Results

- **Test accuracy: 74%**
- Precision, recall and F1-score: **0.74** for both Pass and Fail

### Prediction for a new student

| Study hours | Attendance | Previous marks | Assignment score | Sleep hours |
|-------------|------------|----------------|------------------|-------------|
| 6.0 | 84 | 46 | 80 | 8.0 |

The model predicted **Pass** (about 68% probability for Pass and 32% for Fail).

## Screenshots

Add your screenshots here, for example:

```
![Confusion Matrix](confusion_matrix.png)
```

## How to run

1. Install the libraries:
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
2. Keep `student_performance_dataset.csv` in the same folder as the notebook.
3. Open and run `Student_Performance_ANN_Project.ipynb`:
```
jupyter notebook
```

## What I learned

- Why feature scaling is important for neural networks
- How early stopping helps to avoid overfitting
- How to read a classification report and a confusion matrix
- A model can give probabilities, not only a yes or no answer

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## Future work

- Try more data
- Compare with Logistic Regression and Random Forest
- Tune the hyperparameters to improve accuracy
