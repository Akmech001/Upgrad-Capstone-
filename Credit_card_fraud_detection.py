#!/usr/bin/env python
# coding: utf-8

# In[52]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.stats import randint


# In[53]:


# Load the dataset
df = pd.read_csv('creditcard.csv')


# In[54]:


df.head()


# In[55]:


df.info()


# In[56]:


df.describe()


# In[57]:


# Check the distribution of classes
print(df['Class'].value_counts())


# In[58]:


# Plot histogram of the data
df.hist(bins=30, figsize=(30, 30))
plt.show()


# In[59]:


# Normalize the 'Amount' and 'Time' columns
new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].values.reshape(-1, 1))
new_df['Time'] = (new_df['Time'] - new_df['Time'].min()) / (new_df['Time'].max() - new_df['Time'].min())


# In[60]:


# Shuffle the dataset
new_df = new_df.sample(frac=1, random_state=1)


# In[61]:


# Split the data into training, testing, and validation sets
train, test = train_test_split(new_df, test_size=0.1, random_state=1, stratify=new_df['Class'])
train, val = train_test_split(train, test_size=0.1, random_state=1, stratify=train['Class'])


# In[62]:


# Separate features and target
x_train, y_train = train.drop(columns='Class'), train['Class']
x_test, y_test = test.drop(columns='Class'), test['Class']
x_val, y_val = val.drop(columns='Class'), val['Class']

print(f"Train shape: {x_train.shape}, {y_train.shape}")
print(f"Validation shape: {x_val.shape}, {y_val.shape}")
print(f"Test shape: {x_test.shape}, {y_test.shape}")


# In[63]:


# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[64]:


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


# In[65]:


# Train and evaluate Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)
logistic_train_accuracy = logistic_model.score(x_train, y_train)
logistic_val_predictions = logistic_model.predict(x_val)
print(f"Logistic Regression Train Accuracy: {logistic_train_accuracy}")
print(classification_report(y_val, logistic_val_predictions, target_names=['Not Fraud', 'Fraud']))


# In[66]:


# Confusion matrix and ROC curve for Logistic Regression
cm = confusion_matrix(y_val, logistic_val_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val, logistic_model.predict_proba(x_val)[:, 1])
fpr, tpr, _ = roc_curve(y_val, logistic_model.predict_proba(x_val)[:, 1])
plot_roc_curve(fpr, tpr)


# In[67]:


# Train and evaluate a shallow neural network
shallow_nn = Sequential()
shallow_nn.add(InputLayer(input_shape=(x_train.shape[1],)))
shallow_nn.add(Dense(2, activation='relu'))
shallow_nn.add(BatchNormalization())
shallow_nn.add(Dense(1, activation='sigmoid'))

checkpoint = ModelCheckpoint('shallow_nn.keras', save_best_only=True, monitor='val_loss', mode='min')
shallow_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

shallow_nn.summary()

shallow_nn.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=5, callbacks=[checkpoint])
nn_val_predictions = (shallow_nn.predict(x_val).flatten() > 0.5).astype(int)
print(classification_report(y_val, nn_val_predictions, target_names=['Not Fraud', 'Fraud']))


# In[68]:


# Confusion matrix and ROC curve for Neural Network
cm = confusion_matrix(y_val, nn_val_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val, shallow_nn.predict(x_val).flatten())
fpr, tpr, _ = roc_curve(y_val, shallow_nn.predict(x_val).flatten())
plot_roc_curve(fpr, tpr)


# In[69]:


# Train and evaluate RandomForestClassifier
rf = RandomForestClassifier(max_depth=2, n_jobs=-1, random_state=1)
rf.fit(x_train, y_train)
rf_val_predictions = rf.predict(x_val)
print(classification_report(y_val, rf_val_predictions, target_names=['Not Fraud', 'Fraud']))


# In[70]:


# Confusion matrix and ROC curve for RandomForest
cm = confusion_matrix(y_val, rf_val_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val, rf.predict_proba(x_val)[:, 1])
fpr, tpr, _ = roc_curve(y_val, rf.predict_proba(x_val)[:, 1])
plot_roc_curve(fpr, tpr)


# In[71]:


# Train and evaluate GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=1)
gbc.fit(x_train, y_train)
gbc_val_predictions = gbc.predict(x_val)
print(classification_report(y_val, gbc_val_predictions, target_names=['Not Fraud', 'Fraud']))


# In[72]:


# Confusion matrix and ROC curve for GradientBoosting
cm = confusion_matrix(y_val, gbc_val_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val, gbc.predict_proba(x_val)[:, 1])
fpr, tpr, _ = roc_curve(y_val, gbc.predict_proba(x_val)[:, 1])
plot_roc_curve(fpr, tpr)


# In[73]:


# Train and evaluate LinearSVC
svc = LinearSVC(class_weight='balanced')
svc.fit(x_train, y_train)
svc_val_predictions = svc.predict(x_val)
print(classification_report(y_val, svc_val_predictions, target_names=['Not Fraud', 'Fraud']))


# In[74]:


# Confusion matrix and ROC curve for LinearSVC
cm = confusion_matrix(y_val, svc_val_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
# ROC curve not applicable for LinearSVC as it does not provide probability estimates


# In[75]:


# Addressing imbalance with under-sampling
not_frauds = new_df[new_df['Class'] == 0]
frauds = new_df[new_df['Class'] == 1]
balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])
balanced_df = balanced_df.sample(frac=1, random_state=1)  # Shuffle


# In[76]:


# Split balanced dataset into training, validation, and test sets
train_bal, test_bal = train_test_split(balanced_df, test_size=0.1, random_state=1, stratify=balanced_df['Class'])
train_bal, val_bal = train_test_split(train_bal, test_size=0.1, random_state=1, stratify=train_bal['Class'])

x_train_bal, y_train_bal = train_bal.drop(columns='Class'), train_bal['Class']
x_test_bal, y_test_bal = test_bal.drop(columns='Class'), test_bal['Class']
x_val_bal, y_val_bal = val_bal.drop(columns='Class'), val_bal['Class']

print(f"Balanced Train shape: {x_train_bal.shape}, {y_train_bal.shape}")
print(f"Balanced Validation shape: {x_val_bal.shape}, {y_val_bal.shape}")
print(f"Balanced Test shape: {x_test_bal.shape}, {y_test_bal.shape}")


# In[77]:


# Train and evaluate Logistic Regression on balanced data
logistic_model_bal = LogisticRegression(max_iter=1000)
logistic_model_bal.fit(x_train_bal, y_train_bal)
logistic_val_bal_predictions = logistic_model_bal.predict(x_val_bal)
print(classification_report(y_val_bal, logistic_val_bal_predictions, target_names=['Not Fraud', 'Fraud']))


# In[78]:


# Confusion matrix and ROC curve for Logistic Regression on balanced data
cm = confusion_matrix(y_val_bal, logistic_val_bal_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val_bal, logistic_model_bal.predict_proba(x_val_bal)[:, 1])
fpr, tpr, _ = roc_curve(y_val_bal, logistic_model_bal.predict_proba(x_val_bal)[:, 1])
plot_roc_curve(fpr, tpr)


# In[79]:


# Train and evaluate a shallow neural network on balanced data
shallow_nn_bal = Sequential()
shallow_nn_bal.add(InputLayer(input_shape=(x_train_bal.shape[1],)))
shallow_nn_bal.add(Dense(2, activation='relu'))
shallow_nn_bal.add(BatchNormalization())
shallow_nn_bal.add(Dense(1, activation='sigmoid'))

checkpoint_bal = ModelCheckpoint('shallow_nn_bal.keras', save_best_only=True, monitor='val_loss', mode='min')
shallow_nn_bal.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

shallow_nn_bal.fit(x_train_bal, y_train_bal, validation_data=(x_val_bal, y_val_bal), epochs=40, callbacks=[checkpoint_bal])
nn_val_bal_predictions = (shallow_nn_bal.predict(x_val_bal).flatten() > 0.5).astype(int)
print(classification_report(y_val_bal, nn_val_bal_predictions, target_names=['Not Fraud', 'Fraud']))


# In[80]:


# Confusion matrix and ROC curve for Neural Network on balanced data
cm = confusion_matrix(y_val_bal, nn_val_bal_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val_bal, shallow_nn_bal.predict(x_val_bal).flatten())
fpr, tpr, _ = roc_curve(y_val_bal, shallow_nn_bal.predict(x_val_bal).flatten())
plot_roc_curve(fpr, tpr)


# In[81]:


# Train and evaluate RandomForestClassifier on balanced data
rf_bal = RandomForestClassifier(max_depth=2, n_jobs=-1, random_state=1)
rf_bal.fit(x_train_bal, y_train_bal)
rf_val_bal_predictions = rf_bal.predict(x_val_bal)
print(classification_report(y_val_bal, rf_val_bal_predictions, target_names=['Not Fraud', 'Fraud']))


# In[82]:


# Confusion matrix and ROC curve for RandomForest on balanced data
cm = confusion_matrix(y_val_bal, rf_val_bal_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val_bal, rf_bal.predict_proba(x_val_bal)[:, 1])
fpr, tpr, _ = roc_curve(y_val_bal, rf_bal.predict_proba(x_val_bal)[:, 1])
plot_roc_curve(fpr, tpr)


# In[83]:


# Train and evaluate GradientBoostingClassifier on balanced data
gbc_bal = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=1)
gbc_bal.fit(x_train_bal, y_train_bal)
gbc_val_bal_predictions = gbc_bal.predict(x_val_bal)
print(classification_report(y_val_bal, gbc_val_bal_predictions, target_names=['Not Fraud', 'Fraud']))


# In[84]:


# Confusion matrix and ROC curve for GradientBoosting on balanced data
cm = confusion_matrix(y_val_bal, gbc_val_bal_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val_bal, gbc_bal.predict_proba(x_val_bal)[:, 1])
fpr, tpr, _ = roc_curve(y_val_bal, gbc_bal.predict_proba(x_val_bal)[:, 1])
plot_roc_curve(fpr, tpr)


# In[85]:


# Train and evaluate LinearSVC on balanced data
svc_bal = LinearSVC(class_weight='balanced', max_iter=10000)
svc_bal.fit(x_train_bal, y_train_bal)
svc_val_bal_predictions = svc_bal.predict(x_val_bal)
print(classification_report(y_val_bal, svc_val_bal_predictions, target_names=['Not Fraud', 'Fraud']))


# In[86]:


# Confusion matrix and ROC curve for LinearSVC on balanced data
cm = confusion_matrix(y_val_bal, svc_val_bal_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])


# In[88]:


# Hyperparameter tuning with RandomizedSearchCV for RandomForestClassifier
param_dist_rf = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 6),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

random_search_rf = RandomizedSearchCV(
    RandomForestClassifier(n_jobs=-1, random_state=1),
    param_distributions=param_dist_rf,
    n_iter=20,  # Number of parameter settings that are sampled
    cv=5,
    scoring='accuracy',
    random_state=1
)

random_search_rf.fit(x_train_bal, y_train_bal)

# Get the best estimator and its hyperparameters
best_rf_random = random_search_rf.best_estimator_
print(f"Best RandomForest parameters (random search): {random_search_rf.best_params_}")

# Evaluate the best model on the validation set
rf_val_bal_predictions_random = best_rf_random.predict(x_val_bal)
print(classification_report(y_val_bal, rf_val_bal_predictions_random, target_names=['Not Fraud', 'Fraud']))


# In[89]:


# Confusion matrix and ROC curve for tuned RandomForest
cm = confusion_matrix(y_val_bal, rf_val_bal_predictions_random)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_val_bal, best_rf_random.predict_proba(x_val_bal)[:, 1])
fpr, tpr, _ = roc_curve(y_val_bal, best_rf_random.predict_proba(x_val_bal)[:, 1])
plot_roc_curve(fpr, tpr)


# In[90]:


# Feature importance for RandomForest
feature_importances = best_rf_random.feature_importances_
feature_names = x_train.columns
sorted_idx = np.argsort(feature_importances)


# In[91]:


plt.figure(figsize=(10, 10))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.title('Feature Importance - RandomForest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[96]:


#Steps to Model Deployment
# Save the best model (Logistic Regression with balanced data)
joblib.dump(logistic_model_bal, 'logistic_model_bal.pkl')

# Load the model (for deployment or further use)
loaded_model = joblib.load('logistic_model_bal.pkl')

# Make predictions with the loaded model
test_predictions = loaded_model.predict(x_test_bal)
print(classification_report(y_test_bal, test_predictions, target_names=['Not Fraud', 'Fraud']))


# In[98]:


# Confusion matrix and ROC curve for Logistic Regression on test data
cm = confusion_matrix(y_test_bal, test_predictions)
plot_confusion_matrix(cm, classes=['Not Fraud', 'Fraud'])
roc_auc = roc_auc_score(y_test_bal, loaded_model.predict_proba(x_test_bal)[:, 1])
fpr, tpr, _ = roc_curve(y_test_bal, loaded_model.predict_proba(x_test_bal)[:, 1])
plot_roc_curve(fpr, tpr)


# In[ ]:




