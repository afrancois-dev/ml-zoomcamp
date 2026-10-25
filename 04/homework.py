import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import KFold, train_test_split

    return (
        DictVectorizer,
        KFold,
        LogisticRegression,
        f1_score,
        mo,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(pd):
    df = pd.read_csv("course_lead_scoring_2026.csv")
    target = "converted"
    categorical_features = [
        "lead_source",
        "industry",
        "employment_status",
        "location",
    ]
    numerical_features = [
        "lead_score",
        "number_of_courses_viewed",
        "interaction_count",
        "annual_income",
    ]
    df
    return categorical_features, df, numerical_features, target


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data preparation
    """)
    return


@app.cell
def _(df):
    missing_values = df.isna().sum()
    missing_values
    return


@app.cell
def _(categorical_features, df, numerical_features):
    prepared_df = df.copy()
    prepared_df[categorical_features] = prepared_df[categorical_features].fillna("NA")
    prepared_df[numerical_features] = prepared_df[numerical_features].fillna(0.0)
    remaining_missing_values = int(prepared_df.isna().sum().sum())
    remaining_missing_values
    return (prepared_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Train, validation, and test split
    """)
    return


@app.cell
def _(prepared_df, train_test_split):
    df_full_train, df_test = train_test_split(
        prepared_df, test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )
    return df_full_train, df_train, df_val


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Question 1 - ROC AUC feature importance
    """)
    return


@app.cell
def _(df_train, numerical_features, roc_auc_score, target):
    numerical_auc = {}
    for feature in numerical_features:
        score = roc_auc_score(df_train[target], df_train[feature])
        if score < 0.5:
            score = roc_auc_score(df_train[target], -df_train[feature])
        numerical_auc[feature] = score

    best_numerical_feature = max(numerical_auc, key=numerical_auc.get)
    numerical_auc, best_numerical_feature
    return


@app.cell
def _(DictVectorizer, LogisticRegression, prepared_df, target):
    feature_names = [column for column in prepared_df.columns if column != target]

    def train_model(train_df, eval_df, C=1.0):
        vectorizer = DictVectorizer(sparse=False)
        X_train = vectorizer.fit_transform(
            train_df[feature_names].to_dict(orient="records")
        )
        X_eval = vectorizer.transform(
            eval_df[feature_names].to_dict(orient="records")
        )
        model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
        model.fit(X_train, train_df[target])
        return model.predict_proba(X_eval)[:, 1]

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Question 2 - Validation AUC
    """)
    return


@app.cell
def _(df_train, df_val, roc_auc_score, target, train_model):
    validation_predictions = train_model(df_train, df_val)
    validation_auc = roc_auc_score(df_val[target], validation_predictions)
    round(validation_auc, 3)
    return (validation_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Questions 3 and 4 - Precision, recall, and F1
    """)
    return


@app.cell
def _(
    df_val,
    f1_score,
    pd,
    precision_score,
    recall_score,
    target,
    validation_predictions,
):
    thresholds = [index / 100 for index in range(101)]
    threshold_metrics = []
    for threshold in thresholds:
        predictions = (validation_predictions >= threshold).astype(int)
        precision = precision_score(
            df_val[target], predictions, zero_division=0
        )
        recall = recall_score(df_val[target], predictions, zero_division=0)
        f1 = f1_score(df_val[target], predictions, zero_division=0)
        threshold_metrics.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    metrics_df = pd.DataFrame(threshold_metrics)
    nonzero_metrics = metrics_df.query("precision != 0 or recall != 0")
    precision_recall_threshold = nonzero_metrics.loc[
        (nonzero_metrics["precision"] - nonzero_metrics["recall"])
        .abs()
        .idxmin(),
        "threshold",
    ]
    best_f1_threshold = metrics_df.loc[metrics_df["f1"].idxmax(), "threshold"]
    metrics_df, precision_recall_threshold, best_f1_threshold
    return (metrics_df,)


@app.cell
def _(metrics_df, plt):
    figure, axis = plt.subplots()
    axis.plot(metrics_df["threshold"], metrics_df["precision"], label="precision")
    axis.plot(metrics_df["threshold"], metrics_df["recall"], label="recall")
    axis.set_xlabel("threshold")
    axis.set_ylabel("score")
    axis.legend()
    figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Questions 5 and 6 - 5-fold cross-validation and tuning
    """)
    return


@app.cell
def _(KFold, df_full_train, pd, roc_auc_score, target, train_model):
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    def cross_validate(C=1.0):
        scores = []
        for train_indices, validation_indices in kfold.split(df_full_train):
            fold_train = df_full_train.iloc[train_indices]
            fold_validation = df_full_train.iloc[validation_indices]
            predictions = train_model(fold_train, fold_validation, C=C)
            scores.append(
                roc_auc_score(fold_validation[target], predictions)
            )
        return scores

    cv_scores = cross_validate()
    cv_standard_deviation = pd.Series(cv_scores).std(ddof=0)
    round(cv_standard_deviation, 3)
    return (cross_validate,)


@app.cell
def _(cross_validate, pd):
    hyperparameter_results = []
    for C in [0.000001, 0.001, 1]:
        scores = cross_validate(C)
        hyperparameter_results.append(
            {
                "C": C,
                "mean_auc": pd.Series(scores).mean(),
                "std_auc": pd.Series(scores).std(ddof=0),
            }
        )

    hyperparameter_results = pd.DataFrame(hyperparameter_results)
    best_C = hyperparameter_results.sort_values(
        ["mean_auc", "std_auc", "C"], ascending=[False, True, True]
    ).iloc[0]["C"]
    hyperparameter_results.round(3), best_C
    return


if __name__ == "__main__":
    app.run()
