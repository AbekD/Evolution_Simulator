from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def split_dataset(df):
    X = df.drop(columns=["scenario", "experiment"])
    y = df["scenario"]
    groups = df["experiment"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_index, test_index = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "neutral", "positive"]
    ))


def cross_validate_model(model, X, y, groups):
    cv = GroupKFold(n_splits=5)

    scores = cross_val_score(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="accuracy"
    )

    print("Fold scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())

def show_feature_importance(model, X):
    importance = model.feature_importances_

    for i in range(len(importance)):
        print(X.columns[i], importance[i])