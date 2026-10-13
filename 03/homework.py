import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell
def _(pd):
    raw_df = pd.read_csv("course_lead_scoring_2026.csv")
    target = "converted"
    categorical_features = [
        "lead_source",
        "industry",
        "employment_status",
        "location",
    ]
    numerical_features = [
        "annual_income",
        "number_of_courses_viewed",
        "interaction_count",
        "lead_score",
    ]
    raw_df
    return categorical_features, numerical_features, raw_df, target


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data preparation
    """)
    return


@app.cell
def _(raw_df):
    missing_values = raw_df.isna().sum()
    missing_values
    return (missing_values,)


@app.cell
def _(categorical_features, numerical_features, raw_df):
    df = raw_df.copy()
    df[categorical_features] = df[categorical_features].fillna("NA")
    df[numerical_features] = df[numerical_features].fillna(0.0)
    df.isna().sum()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1 - What is the mode for `industry`?
    """)
    return


@app.cell
def _(df):
    industry_mode = df["industry"].mode().iloc[0]
    industry_mode
    return (industry_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2 - Which numerical features have the biggest correlation?
    """)
    return


@app.cell
def _(df, numerical_features):
    correlation_matrix = df[numerical_features].corr()
    correlation_matrix
    return (correlation_matrix,)


@app.cell
def _(correlation_matrix):
    correlation_options = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    correlation_scores = {
        pair: correlation_matrix.loc[pair[0], pair[1]]
        for pair in correlation_options
    }
    most_correlated_features = max(
        correlation_scores, key=correlation_scores.get
    )
    correlation_scores, most_correlated_features
    return correlation_scores, most_correlated_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split the data
    """)
    return


@app.cell
def _(df, target):
    from sklearn.model_selection import train_test_split

    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=42
    )

    y_train = df_train[target]
    y_val = df_val[target]
    X_train = df_train.drop(columns=target)
    X_val = df_val.drop(columns=target)
    X_train, X_val, y_train, y_val
    return X_train, X_val, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 3 - Which categorical feature has the biggest mutual information?
    """)
    return


@app.cell
def _(X_train, categorical_features, y_train):
    from sklearn.metrics import mutual_info_score

    mutual_information_scores = {
        feature: round(mutual_info_score(y_train, X_train[feature]), 2)
        for feature in categorical_features
    }
    most_informative_feature = max(
        mutual_information_scores, key=mutual_information_scores.get
    )
    mutual_information_scores, most_informative_feature
    return mutual_information_scores, most_informative_feature


@app.cell
def _(pd):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    def evaluate_logistic_regression(
        X_train,
        y_train,
        X_eval,
        y_eval,
        categorical_features,
        C=1.0,
    ):
        X_train_encoded = pd.get_dummies(
            X_train, columns=categorical_features, dtype=int
        )
        X_eval_encoded = pd.get_dummies(
            X_eval, columns=categorical_features, dtype=int
        ).reindex(columns=X_train_encoded.columns, fill_value=0)

        model = LogisticRegression(
            solver="liblinear", C=C, max_iter=1000, random_state=42
        )
        model.fit(X_train_encoded, y_train)
        accuracy = accuracy_score(y_eval, model.predict(X_eval_encoded))
        return accuracy

    return (evaluate_logistic_regression,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 4 - What is the validation accuracy?
    """)
    return


@app.cell
def _(
    X_train,
    X_val,
    categorical_features,
    evaluate_logistic_regression,
    y_train,
    y_val,
):
    accuracy = evaluate_logistic_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        categorical_features,
    )
    rounded_accuracy = round(accuracy, 2)
    rounded_accuracy
    return accuracy, rounded_accuracy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 5 - Which feature has the smallest accuracy difference when removed?
    """)
    return


@app.cell
def _(
    X_train,
    X_val,
    accuracy,
    categorical_features,
    evaluate_logistic_regression,
    y_train,
    y_val,
):
    features_to_eliminate = [
        "lead_source",
        "number_of_courses_viewed",
        "interaction_count",
    ]
    accuracy_differences = {}
    for feature in features_to_eliminate:
        remaining_categorical_features = [
            column
            for column in categorical_features
            if column != feature
        ]
        accuracy_without_feature = evaluate_logistic_regression(
            X_train.drop(columns=feature),
            y_train,
            X_val.drop(columns=feature),
            y_val,
            remaining_categorical_features,
        )
        accuracy_differences[feature] = accuracy - accuracy_without_feature

    least_useful_feature = min(
        accuracy_differences, key=accuracy_differences.get
    )
    accuracy_differences, least_useful_feature
    return accuracy_differences, least_useful_feature


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 6 - Which regularization value gives the best accuracy?
    """)
    return


@app.cell
def _(
    X_train,
    X_val,
    categorical_features,
    evaluate_logistic_regression,
    y_train,
    y_val,
):
    regularization_values = [0.000001, 0.00001, 0.0001, 0.001]
    accuracy_by_C = {
        C: round(
            evaluate_logistic_regression(
                X_train,
                y_train,
                X_val,
                y_val,
                categorical_features,
                C=C,
            ),
            3,
        )
        for C in regularization_values
    }
    best_C = max(regularization_values, key=lambda C: (accuracy_by_C[C], -C))
    accuracy_by_C, best_C
    return accuracy_by_C, best_C


@app.cell(hide_code=True)
def _(
    accuracy_differences,
    best_C,
    industry_mode,
    most_correlated_features,
    most_informative_feature,
    mo,
    rounded_accuracy,
    least_useful_feature,
):
    mo.md(f"""
    **Answers**

    1. `industry` mode: **{industry_mode}**
    2. Most correlated pair: **{most_correlated_features[0]}** and **{most_correlated_features[1]}**
    3. Highest mutual information: **{most_informative_feature}**
    4. Validation accuracy: **{rounded_accuracy}**
    5. Smallest accuracy difference: **{least_useful_feature}** (`{accuracy_differences[least_useful_feature]:.3f}`)
    6. Best regularization parameter: **{best_C}**
    """)
    return


if __name__ == "__main__":
    app.run()
