# =========================================================
# 🔥 COMPLETE ML LAB - TRUE FINAL (ALL 12 QUESTIONS)
# =========================================================
"""

# =========================================================
# Q1: PCA + CLASSIFICATION
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col = df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

y_iloc = df.iloc[:, -1]
X_iloc = df.iloc[:, :-1]

X = pd.get_dummies(X, drop_first=True)
if y.dtype=='object': y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)
X = PCA(n_components=0.95).fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestClassifier().fit(X_train,y_train)
y_pred = model.predict(X_test)

print("\nQ1 PCA\nAccuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q2: ENSEMBLE
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col = df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

y_iloc = df.iloc[:, -1]
X_iloc = df.iloc[:, :-1]

X = pd.get_dummies(X, drop_first=True)
if y.dtype=='object': y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf = RandomForestClassifier()
svm = SVC(probability=True)
knn = KNeighborsClassifier()

ensemble = VotingClassifier([('rf',rf),('svm',svm),('knn',knn)], voting='soft')
ensemble.fit(X_train,y_train)

y_pred = ensemble.predict(X_test)

print("\nQ2 Ensemble\nAccuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q3: SVM
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col = df.columns[-1]
y = df[target_col]
X = df.drop(columns=[target_col])

y_iloc = df.iloc[:, -1]
X_iloc = df.iloc[:, :-1]

X = pd.get_dummies(X, drop_first=True)
if y.dtype=='object': y = LabelEncoder().fit_transform(y)

X = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for k in ['linear','rbf','poly']:
    model = SVC(kernel=k, probability=True).fit(X_train,y_train)
    y_pred = model.predict(X_test)

    print(f"\nSVM {k}\nAccuracy:",accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q4: CLUSTERING
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

try:
    from kmodes.kmodes import KModes
    kmodes_available=True
except:
    kmodes_available=False

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.iloc[:, :-1]
X_iloc = df.iloc[:, :-1]

X = pd.get_dummies(X)
X = StandardScaler().fit_transform(X)

# Elbow
wcss=[KMeans(i).fit(X).inertia_ for i in range(1,10)]
plt.plot(range(1,10),wcss); plt.title("Elbow"); plt.show()

# Silhouette
for k in range(2,5):
    print("Silhouette:",silhouette_score(X,KMeans(k).fit_predict(X)))

# PCA for visualization
X2 = PCA(n_components=2).fit_transform(X)

plt.scatter(X2[:,0],X2[:,1],c=KMeans(3).fit_predict(X)); plt.title("KMeans"); plt.show()
plt.scatter(X2[:,0],X2[:,1],c=DBSCAN().fit_predict(X)); plt.title("DBSCAN"); plt.show()
plt.scatter(X2[:,0],X2[:,1],c=AgglomerativeClustering(3).fit_predict(X)); plt.title("Hierarchical"); plt.show()

sch.dendrogram(sch.linkage(X,'ward')); plt.show()

if kmodes_available:
    print("K-Modes:", KModes(3).fit_predict(df.astype(str))[:10])


# =========================================================
# Q5: PERCEPTRON
# =========================================================
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=Perceptron().fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q6: MLP
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=MLPClassifier(max_iter=500).fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()

plt.plot(model.loss_curve_); plt.title("Loss Curve"); plt.show()


# =========================================================
# Q7: LOGISTIC REGRESSION
# =========================================================
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000).fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q8: KNN
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

acc=[]
for k in range(1,10):
    acc.append(KNeighborsClassifier(k).fit(X_train,y_train).score(X_test,y_test))

plt.plot(range(1,10),acc); plt.show()

model=KNeighborsClassifier(5).fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q9: NAIVE BAYES
# =========================================================
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X=StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=GaussianNB().fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()


# =========================================================
# Q10: SIMPLE LINEAR REGRESSION
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

y=df.iloc[:,-1]
X=df.iloc[:,[0]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression().fit(X_train,y_train)
y_pred=model.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred))

plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,color='red')
plt.show()


# =========================================================
# Q11: MULTIPLE LINEAR REGRESSION
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

target_col=df.columns[-1]
y=df[target_col]; X=pd.get_dummies(df.drop(columns=[target_col]))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression().fit(X_train,y_train)
y_pred=model.predict(X_test)

print("MSE:",mean_squared_error(y_test,y_pred))

plt.scatter(y_pred,y_test-y_pred); plt.axhline(0,color='red'); plt.show()
plt.scatter(y_test,y_pred); plt.show()


# =========================================================
# Q12: DECISION TREE (CLASSIFICATION + REGRESSION)
# =========================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import *

df = pd.read_csv("sample.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

# classification
target_col=df.columns[-1]
y=df[target_col]; X=df.drop(columns=[target_col])

y_iloc=df.iloc[:,-1]; X_iloc=df.iloc[:,:-1]

X=pd.get_dummies(X)
if y.dtype=='object': y=LabelEncoder().fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

clf=DecisionTreeClassifier(max_depth=5).fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("DT Accuracy:",accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True); plt.show()

plt.figure(figsize=(10,5))
plot_tree(clf,filled=True); plt.show()

# regression
y_reg=df.iloc[:,-1]
if y_reg.dtype!='object':
    X_reg=pd.get_dummies(df.drop(columns=[target_col]))
    X_train,X_test,y_train,y_test=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)

    reg=DecisionTreeRegressor(max_depth=5).fit(X_train,y_train)
    y_pred=reg.predict(X_test)

    print("DT MSE:",mean_squared_error(y_test,y_pred))

    plt.scatter(y_test,y_pred); plt.show()
    plt.scatter(y_pred,y_test-y_pred); plt.axhline(0,color='red'); plt.show()

#CLUSTERING ASSESSMENT QSTN 
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# LOAD DATA
df = pd.read_csv('D6_train.csv') # [cite: 54, 83]
X_scaled = StandardScaler().fit_transform(df)

# APPLY CLUSTERING
# 
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5)
db_labels = dbscan.fit_predict(X_scaled)

# EVALUATE PERFORMANCE
# [cite: 62, 91]
hc_score = silhouette_score(X_scaled, hc_labels)
print(f"Hierarchical Silhouette Score: {hc_score:.4f}")
# INFERENCE: 
# Score > 0.5: Well-defined, separate clusters.
# Score < 0.2: Significant overlapping or poor cluster structure.

# IDENTIFY BEST MODEL
# [cite: 59, 88]
best_labels = hc_labels if hc_score > 0 else db_labels

#PCA ASSESSMENT QSTN
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB # For Q2 [cite: 41]
from sklearn.svm import SVC                 # For Q6 [cite: 109]
from sklearn.metrics import accuracy_score

# LOAD AND SCALE (PCA requires scaled data)
df = pd.read_csv('D5_train.csv') 
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. APPLY PCA
# [cite: 40, 107]
pca = PCA(n_components=0.95) # Retain 95% of variance
X_pca = pca.fit_transform(X_scaled)
print(f"Features reduced from {X.shape[1]} to {X_pca.shape[1]}")
# INFERENCE: PCA simplifies the model by merging highly correlated features.

# 2. COMPARE PERFORMANCE
# Q2 uses Naive Bayes[cite: 41, 42]; Q6 uses SVM [cite: 109, 110]
model = GaussianNB() # Swap with SVC(kernel='rbf') for Q6
model.fit(X_pca, y)
acc = accuracy_score(y, model.predict(X_pca))
print(f"Accuracy with PCA features: {acc:.2%}")
# INFERENCE: If accuracy drops significantly, the dropped components held vital info.

# 9. FINAL PREDICTION ON TEST SET
# [cite: 47, 115]
test_data = pd.read_csv('D5_test_noTarget.csv')
test_scaled = scaler.transform(test_data)
test_pca = pca.transform(test_scaled)
final_preds = model.predict(test_pca)

#ENSEMBLE ASSESSMENT QSTN
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score

# 1. LOAD DATASET (Change filename based on your question: D5_train or D4_train)
# [cite: 15, 66]
df = pd.read_csv('D5_train.csv') 

# 2. EXPLORATORY DATA ANALYTICS & VISUALIZATION
# [cite: 16, 17, 67, 68]
print("--- Dataset Description ---")
print(df.describe()) # INFERENCE: Check for scaling needs or outliers

# 3. PRE-PROCESSING
# [cite: 18, 69]
df.fillna(df.mean(), inplace=True) # INFERENCE: Handling missing values to prevent model errors
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. CORRELATION ANALYSIS
# [cite: 19, 70]
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show() 
# INFERENCE: Values close to 1.0 or -1.0 indicate highly redundant features.

# 5. ENSEMBLE ALGORITHM (Majority Voting)
# 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestClassifier(n_estimators=100)
model2 = AdaBoostClassifier(n_estimators=100)
ensemble = VotingClassifier(estimators=[('rf', model1), ('ada', model2)], voting='hard')
ensemble.fit(X_train, y_train)

# 7. TABULATE RESULTS & CONFUSION MATRIX
# [cite: 22, 73]
y_pred = ensemble.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
print("--- Confusion Matrix ---")
print(cm) 
# INFERENCE: The diagonal shows correct predictions; off-diagonal shows errors.

print("\n--- Classification Report ---")
print(classification_report(y_val, y_pred))
# INFERENCE: F1-Score represents the balance between Precision and Recall.

# 9. PREDICT ON UNLABELLED TEST SET
# [cite: 25, 76]
test_df = pd.read_csv('D5_test_noTarget.csv') 
test_predictions = ensemble.predict(test_df)
output = pd.DataFrame({'Serial Number': range(1, len(test_predictions)+1), 
                       'Predicted Label': test_predictions})
output.to_csv('predictions.csv', index=False)

    
"""
