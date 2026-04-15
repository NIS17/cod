PCA
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = df.iloc[:, :-1]

X = pd.get_dummies(X, drop_first=True)
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)
pca = PCA(n_components=0.95)
X = pca.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()

print("\n===== INFERENCE =====")
print("PCA reduced dimensions while keeping 95% variance.")
print("Helps remove redundant features and improves efficiency.")

Ensemble 
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = VotingClassifier([
    ('rf',RandomForestClassifier()),
    ('svm',SVC(probability=True)),
    ('knn',KNeighborsClassifier())
], voting='soft')

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

sns.heatmap(confusion_matrix(y_test,y_pred),annot=True)
plt.show()

print("\n===== INFERENCE =====")
print("Ensemble combines multiple models for better accuracy.")
print("Soft voting improves prediction using probabilities.")

Svm
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for k in ['linear','rbf','poly']:
    model = SVC(kernel=k).fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print(f"\n{k} Accuracy:",accuracy_score(y_test,y_pred))

print("\n===== INFERENCE =====")
print("Different kernels handle different data patterns.")
print("RBF usually performs best for complex data.")

Clustering
import pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X = pd.get_dummies(df.iloc[:, :-1])
X = StandardScaler().fit_transform(X)

# Elbow
wcss = [KMeans(i).fit(X).inertia_ for i in range(1,6)]
plt.plot(range(1,6),wcss)
plt.show()

# Silhouette
score = silhouette_score(X,KMeans(3).fit_predict(X))
print("Silhouette:",score)

# PCA visualization
X2 = PCA(2).fit_transform(X)
plt.scatter(X2[:,0],X2[:,1],c=KMeans(3).fit_predict(X))
plt.show()

print("\n===== INFERENCE =====")
print("Elbow method finds optimal clusters.")
print("Silhouette score shows clustering quality.")

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Perceptron().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\n===== INFERENCE =====")
print("Perceptron works for linearly separable data.")

Logistic
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LogisticRegression(max_iter=1000).fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\n===== INFERENCE =====")
print("Logistic regression is used for classification problems.")

Knn
import pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

acc=[]
for k in range(1,10):
    acc.append(KNeighborsClassifier(k).fit(X_train,y_train).score(X_test,y_test))

plt.plot(range(1,10),acc)
plt.show()

print("\n===== INFERENCE =====")
print("Best K gives highest accuracy.")

Naive Bayes
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = GaussianNB().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\n===== INFERENCE =====")
print("Naive Bayes assumes independent features.")

Simple lr
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")

y = df.iloc[:, -1]
X = df.iloc[:, [0]]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred))

plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred)
plt.show()

print("\n===== INFERENCE =====")
print("Simple linear regression fits a straight line.")

11 morning
import pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred))

plt.scatter(y_test,y_pred)
plt.show()

print("\n===== INFERENCE =====")
print("Multiple regression uses multiple features.")

Decisiontree
import pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")

y = df.iloc[:, -1]
X = pd.get_dummies(df.iloc[:, :-1])

if y.dtype=='object':
    y = LabelEncoder().fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = DecisionTreeClassifier(max_depth=5).fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

print("\n===== INFERENCE =====")
print("Decision tree is easy to interpret but may overfit.")

Clustering assign
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# LOAD DATA
df = pd.read_csv('D6_train.csv')

# SCALE DATA
X_scaled = StandardScaler().fit_transform(df)

# APPLY CLUSTERING
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

# EVALUATE
hc_score = silhouette_score(X_scaled, hc_labels)
print("Hierarchical Silhouette Score:", hc_score)

# INFERENCE
print("\n===== INFERENCE =====")
if hc_score > 0.5:
    print("Clusters are well separated and clearly defined.")
elif hc_score < 0.2:
    print("Clusters are poorly formed with high overlap.")
else:
    print("Clusters are moderately separated.")

print("Agglomerative clustering groups data hierarchically.")
print("DBSCAN detects noise and outliers effectively.")
print("Best model is selected based on silhouette score.")

Assessment pca
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# LOAD DATA
df = pd.read_csv('D5_train.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# SCALE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# APPLY PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"Original Features: {X.shape[1]}")
print(f"Reduced Features: {X_pca.shape[1]}")

# MODEL
model = GaussianNB()
model.fit(X_pca, y)

y_pred = model.predict(X_pca)
acc = accuracy_score(y, y_pred)

print("Accuracy:", acc)

# TEST DATA PREDICTION
test_data = pd.read_csv('D5_test_noTarget.csv')
test_scaled = scaler.transform(test_data)
test_pca = pca.transform(test_scaled)

final_preds = model.predict(test_pca)

# SAVE OUTPUT
output = pd.DataFrame({
    'Serial Number': range(1, len(final_preds)+1),
    'Predicted Label': final_preds
})
output.to_csv('pca_predictions.csv', index=False)

# INFERENCE
print("\n===== INFERENCE =====")
print("PCA reduced dimensionality while retaining 95% variance.")
print("Reduced features improve model efficiency and reduce overfitting.")
print("If accuracy is high, important information is preserved.")
print("If accuracy drops, some important features were lost.")

Ensemble assessment 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report

# LOAD DATA
df = pd.read_csv('D5_train.csv')

# PREPROCESSING
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# CORRELATION
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# SPLIT
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODELS
rf = RandomForestClassifier(n_estimators=100)
ada = AdaBoostClassifier(n_estimators=100)

ensemble = VotingClassifier(
    estimators=[('rf', rf), ('ada', ada)],
    voting='hard'
)

ensemble.fit(X_train, y_train)

# PREDICTION
y_pred = ensemble.predict(X_val)

# RESULTS
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# TEST DATA
test_df = pd.read_csv('D5_test_noTarget.csv')
test_predictions = ensemble.predict(test_df)

output = pd.DataFrame({
    'Serial Number': range(1, len(test_predictions)+1),
    'Predicted Label': test_predictions
})
output.to_csv('ensemble_predictions.csv', index=False)

# INFERENCE
print("\n===== INFERENCE =====")
print("Ensemble learning combines multiple models to improve performance.")
print("Random Forest reduces variance, AdaBoost reduces bias.")
print("Voting classifier improves overall accuracy.")
print("Confusion matrix shows prediction correctness.")
print("Better performance than individual models.")