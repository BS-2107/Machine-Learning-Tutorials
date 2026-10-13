import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop("id", axis=1)

X = df.drop("stroke", axis=1)
y = df["stroke"]

# Identify columns
num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and compare SVM kernels
for kernel in ["linear", "rbf", "sigmoid"]:
    model = Pipeline([
        ("prep", preprocessor),
        ("svm", SVC(kernel=kernel, class_weight="balanced"))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(kernel, "Accuracy:", round(
        accuracy_score(y_test, pred) * 100, 2
    ), "%")