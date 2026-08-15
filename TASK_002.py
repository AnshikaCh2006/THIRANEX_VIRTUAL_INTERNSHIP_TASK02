import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

file_path = r"TASK 01/archive/marketing_campaign.csv"

df = pd.read_csv(file_path, sep="\t")

df["Income"] = df["Income"].fillna(df["Income"].median())

df["Age"] = 2026 - df["Year_Birth"]

df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

df["Total_Spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
)

df["Total_Purchases"] = (
    df["NumWebPurchases"] +
    df["NumCatalogPurchases"] +
    df["NumStorePurchases"]
)

df = df[
    (df["Age"] >= 18) &
    (df["Age"] <= 100)
]

df = df.drop_duplicates()

df["Response"] = df["Response"].astype(int)

features = [
    "Income",
    "Age",
    "Total_Children",
    "Total_Spending",
    "Total_Purchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5"
]

features = [column for column in features if column in df.columns]

X = df[features]
y = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

decision_tree_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

results = {}

for name, model in models.items():

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Predictions": predictions,
        "Probabilities": probabilities
    }

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

results_table = pd.DataFrame({
    model: {
        "Accuracy": results[model]["Accuracy"],
        "Precision": results[model]["Precision"],
        "Recall": results[model]["Recall"],
        "F1 Score": results[model]["F1 Score"]
    }
    for model in results
}).T

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_table)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_table.reset_index(),
    x="index",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)

for ax, (name, result) in zip(axes, results.items()):

    cm = confusion_matrix(
        y_test,
        result["Predictions"]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))

for name, result in results.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        result["Probabilities"]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_auc:.2f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

best_model = results_table["F1 Score"].idxmax()

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best model:", best_model)
print(
    "Best F1 Score:",
    round(results_table.loc[best_model, "F1 Score"], 4)
)

print(
    "Best Accuracy:",
    round(results_table.loc[best_model, "Accuracy"], 4)
)

if best_model == "Logistic Regression":
    final_model = logistic_model
    feature_importance = pd.Series(
        np.abs(logistic_model.coef_[0]),
        index=features
    ).sort_values(ascending=False)
else:
    final_model = models[best_model]
    feature_importance = pd.Series(
        final_model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

print("\nTop Important Features:")
print(feature_importance.head(10))

plt.figure(figsize=(10, 6))

sns.barplot(
    x=feature_importance.head(10).values,
    y=feature_importance.head(10).index
)

plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

results_table.to_csv(
    "model_performance.csv"
)

print("\nModel performance saved as model_performance.csv")

print("\nTask 02 completed successfully.")