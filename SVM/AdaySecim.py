import numpy as np
import matplotlib.pyplot as plt
from faker import Faker
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


fake = Faker()
np.random.seed(42)

n_samples = 300
experience_year = np.random.uniform(0, 10, n_samples)
technical_score = np.random.uniform(0, 100, n_samples)

labels = []

# Değişken isimlerini değiştirdim
for exp, score in zip(experience_year, technical_score):
    if exp < 2 and score > 60:
        labels.append(1)  # İşe alınmadı (başarısız aday)
    else:
        labels.append(0)  # İşe alındı (başarılı aday)

X = np.column_stack((experience_year, technical_score))
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

def predict(experience_year, technical_score):
    prediction_data = np.array([[experience_year, technical_score]])
    scaled_prediction_data = scaler.transform(prediction_data)
    prediction = model.predict(scaled_prediction_data)[0]
    if prediction == 1:
        print("Başarısız aday")
    else:
        print("Başarılı aday")

predict(1.2, 25)
predict(2.5, 55)
predict(1.5, 40)
predict(3.5, 85)
predict(4, 90)

def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=60, edgecolors='k', alpha=0.7)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=150, linewidth=1.5, facecolors='none', edgecolors='k')

    plt.title("Faker Verisiyle SVM: Kredi Riski Tahmini")
    plt.xlabel("Deneyim Yılı (standardize)")
    plt.ylabel("Teknik Puan (standardize)")
    plt.grid(True)
    plt.show()

plot_decision_boundary(model, X_scaled, y)
