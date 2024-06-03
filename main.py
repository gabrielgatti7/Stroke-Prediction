import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.over_sampling import SMOTE

# ===================== Reading data =====================
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# ===================== Data Exploration =====================
# y = data['stroke']
# print(f'Percentage of patient had a stroke: % {round(y.value_counts(normalize=True)[1]*100,2)} --> ({y.value_counts()[1]} patient)\nPercentage of patient did not have a stroke: % {round(y.value_counts(normalize=True)[0]*100,2)} --> ({y.value_counts()[0]} patient)')
# Thus, we have a highly imbalanced dataset

# ===================== Handling missing data =====================
# Plot of missing values
# plt.title('Missing Value Status',fontweight='bold')
# ax = sns.heatmap(data.isna().sum().to_frame(),annot=True,fmt='d',cmap='vlag')
# ax.set_xlabel('Amount Missing')
# plt.show()

# Handling missing bmi values by filling them with the mean of the column
# data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

# A more interesting way is to handle missing bmi values by predicting them based on age and gender
# from: https://www.kaggle.com/code/thomaskonstantin/analyzing-and-modeling-stroke-data
DT_bmi_pipe = Pipeline( steps=[
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
data['gender'] = data['gender'].astype('category')
X = data[['age','gender','bmi']].copy()
X.gender = X.gender.cat.rename_categories({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)

Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
data.loc[Missing.index,'bmi'] = predicted_bmi


# ===================== Optimizing data types =====================
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
data[['hypertension', 'heart_disease']] = data[['hypertension', 'heart_disease']].astype(np.uint8)
data['ever_married'] = data['ever_married'].map({'No': 0, 'Yes': 1}).astype(np.uint8)
data['work_type'] = data['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)
data['Residence_type'] = data['Residence_type'].map({'Rural': 0, 'Urban': 1}).astype(np.uint8)
data['smoking_status'] = data['smoking_status'].map({'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': -1}).astype(np.uint8)
# data.info()


# ===================== Separating data =====================
X = data.drop(['id', 'stroke'], axis=1)

y = data['stroke']

# Using SMOTE to deal with imbalanced data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train)


# ===================== Creating pipelines =====================
numeric_features = ['age', 'avg_glucose_level', 'bmi']
numeric_transformer = Pipeline(steps=[
    ("normalization", MaxAbsScaler()),
    # scales features such as the maximum absolute value is 1 (so the data is guaranteed to be in a [-1, 1] range)
    ('scaler', StandardScaler())
    # standardize features making the mean equal to 0 and variance equal to 1.
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])
# ColumnTransformer allows to apply specific transformations to specific columns of the dataset.


# ===================== Model Choice and Training =====================
knn = Pipeline(steps=[('preprocessor', preprocessor),
                     ('classifier', KNeighborsClassifier(n_neighbors=5))
                      ])

# Create a StratifiedKFold object
stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
accuracy = cross_val_score(knn, X_train_resh, y_train_resh, cv=stratified_kfold)
recall = cross_val_score(knn, X_train_resh, y_train_resh, cv=stratified_kfold, scoring='recall')
precision = cross_val_score(knn, X_train_resh, y_train_resh, cv=stratified_kfold, scoring='precision')
f1s = cross_val_score(knn, X_train_resh, y_train_resh, cv=stratified_kfold, scoring='f1_macro')
roc = cross_val_score(knn, X_train_resh, y_train_resh, cv=stratified_kfold, scoring='roc_auc')



# ===================== Model Assessment =====================
# Print the mean and standard deviation of the scores
# Accuracy: is the number of hits in our model divided by the total sample.
print(f"Accuracy: {accuracy.mean():.2f} (+/- {accuracy.std() * 2:.2f})")

# Precision: of all data classified as positive, how many are actually positive.
# (a precision of 1.0 means that there were no false positives.)
print(f"Precision: {precision.mean():.2f} (+/- {precision.std() * 2:.2f})")

# Recall: what percentage of data is classified as positive compared to the actual 
# number of positives that exist in our sample.
# (a recall of 1.0 means that there were no false negatives.)
print(f"Recall: {recall.mean():.2f} (+/- {recall.std() * 2:.2f})")

# F1-score: this metric combines precision and recall in order to provide a single 
# number that determines the general quality of our model.
print(f"F1-macro: {f1s.mean():.2f} (+/- {f1s.std() * 2:.2f})")
print(f"ROC: {roc.mean():.2f} (+/- {roc.std() * 2:.2f})")
