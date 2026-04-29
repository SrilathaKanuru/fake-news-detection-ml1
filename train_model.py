import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# load cleaned dataset
data = pd.read_csv('cleaned_news.csv')

# input and output
X = data['cleaned_text']
y = data['label']

# convert text to numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# train and test models
for name, model in models.items():
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    # confusion matrix
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()
plt.title(f"Confusion Matrix - {name}")

plt.show()
    
    print(f"{name} Accuracy: {accuracy:.2f}")
    import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("cleaned_news.csv")

# input and output
X = data["cleaned_text"]
y = data["label"]

# convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# store results
model_names = []
accuracies = []

# train models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2f}")

    model_names.append(name)
    accuracies.append(accuracy)

# plot accuracy graph
plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Machine Learning Model Comparison")

plt.xticks(rotation=10)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("cleaned_news.csv")

# input and output
X = data["cleaned_text"]
y = data["label"]

# convert text into numbers
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# store results
model_names = []
accuracies = []

# train models
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.2f}")

    model_names.append(name)
    accuracies.append(accuracy)

# plot accuracy graph
plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Machine Learning Model Comparison")

plt.xticks(rotation=10)

plt.show()
