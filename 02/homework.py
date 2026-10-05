import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    return mo, np, pd


@app.cell
def _(pd):
    df = pd.read_csv("car_fuel_efficiency_2026.csv")
    df = df[['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']]
    df
    return (df,)


@app.cell
def _(df):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.histplot(df["fuel_efficiency_mpg"], bins=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1 - There's one column with missing values. What is it?
    """)
    return


@app.cell
def _(df):
    df.isnull().sum()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2 - What's the median (50% percentile) for variable 'horsepower'?
    """)
    return


@app.cell(hide_code=True)
def _(df):
    df["horsepower"].median()
    return


@app.cell
def _(np):
    def train_linear_regression(X, y):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv.dot(X.T).dot(y)

        return w_full[0], w_full[1:]

    return (train_linear_regression,)


@app.cell
def _(np):
    def rmse(y, y_pred):
        se = (y - y_pred) ** 2
        mse = se.mean()
        return np.sqrt(mse)

    return (rmse,)


@app.cell
def _(np, pd):
    def prepare_validation_framework(df: pd.DataFrame, seed: int, fill_method: str):
        n = len(df)
        n_val = int(n * 0.2)
        n_test = int(n * 0.2)
        n_train = n - n_val - n_test

        np.random.seed(seed)
        idx = np.arange(n)
        np.random.shuffle(idx)

        df_train = df.iloc[idx[:n_train]]
        df_val = df.iloc[idx[n_train:n_train + n_val]]
        df_test = df.iloc[idx[n_train + n_val:]]

        df_train = df_train.copy()
        df_val = df_val.copy()
        df_test = df_test.copy()

        if fill_method == "with_0":
            fill_value = 0
        else:
            fill_value = df_train["horsepower"].mean()

        for split in (df_train, df_val, df_test):
            split["horsepower"] = split["horsepower"].fillna(fill_value)

        return df_train, df_val, df_test

    return (prepare_validation_framework,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### with 0
    """)
    return


@app.cell
def _(df, prepare_validation_framework, rmse, train_linear_regression):
    df_train_0, df_val_0, _ = prepare_validation_framework(df, seed=42, fill_method="with_0")
    X_train_0 = df_train_0.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_train_0 = df_train_0["fuel_efficiency_mpg"].to_numpy()
    X_val_0 = df_val_0.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_val_0 = df_val_0["fuel_efficiency_mpg"].to_numpy()

    intercept_0, weights_0 = train_linear_regression(X_train_0, y_train_0)
    y_pred_0 = intercept_0 + X_val_0.dot(weights_0)
    rmse_with_0 = round(rmse(y_val_0, y_pred_0), 3)
    rmse_with_0
    return (rmse_with_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### with mean
    """)
    return


@app.cell
def _(df, prepare_validation_framework, rmse, train_linear_regression):
    df_train_mean, df_val_mean, _ = prepare_validation_framework(df, seed=42, fill_method="with_mean")
    X_train_mean = df_train_mean.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_train_mean = df_train_mean["fuel_efficiency_mpg"].to_numpy()
    X_val_mean = df_val_mean.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_val_mean = df_val_mean["fuel_efficiency_mpg"].to_numpy()

    intercept_mean, weights_mean = train_linear_regression(X_train_mean, y_train_mean)
    y_pred_mean = intercept_mean + X_val_mean.dot(weights_mean)
    rmse_with_mean = round(rmse(y_val_mean, y_pred_mean), 3)
    rmse_with_mean
    return (rmse_with_mean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Question 3 - Which option gives better RMSE?
    """)
    return


@app.cell
def _(mo, rmse_with_0, rmse_with_mean):
    better_option = "With 0" if rmse_with_0 < rmse_with_mean else "With mean" if rmse_with_mean < rmse_with_0 else "Both are equally good"
    mo.md(f"RMSE with 0: **{rmse_with_0}**  \nRMSE with mean: **{rmse_with_mean}**  \nBetter option: **{better_option}**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 4 - Which r gives the best RMSE?
    """)
    return


@app.cell
def _(np):
    def train_linear_regression_reg(X, y, r=0.001):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX = XTX + r * np.eye(XTX.shape[0])

        XTX_inv = np.linalg.inv(XTX)
        w_full = XTX_inv.dot(X.T).dot(y)

        return w_full[0], w_full[1:]

    return (train_linear_regression_reg,)


@app.cell
def _(df, prepare_validation_framework, rmse, train_linear_regression_reg):
    df_train_reg, df_val_reg, _ = prepare_validation_framework(df, seed=42, fill_method="with_0")
    X_train_reg = df_train_reg.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_train_reg = df_train_reg["fuel_efficiency_mpg"].to_numpy()
    X_val_reg = df_val_reg.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_val_reg = df_val_reg["fuel_efficiency_mpg"].to_numpy()

    regularization_values = [0, 0.01, 0.1, 1, 5, 10, 100]
    rmse_by_r = {}
    for regularization_value in regularization_values:
        intercept_reg, weights_reg = train_linear_regression_reg(
            X_train_reg, y_train_reg, r=regularization_value
        )
        y_pred_reg = intercept_reg + X_val_reg.dot(weights_reg)
        rmse_by_r[regularization_value] = round(rmse(y_val_reg, y_pred_reg), 4)

    best_r = min(regularization_values, key=lambda value: (rmse_by_r[value], value))
    rmse_by_r, best_r
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 5 - How does the seed influence the score?
    """)
    return


@app.cell
def _(df, np, prepare_validation_framework, rmse, train_linear_regression):
    seed_values = list(range(10))
    rmse_by_seed = {}

    for seed_value in seed_values:
        df_train_seed, df_val_seed, _ = prepare_validation_framework(df, seed=seed_value, fill_method="with_0")
        X_train_seed = df_train_seed.drop(columns="fuel_efficiency_mpg").to_numpy()
        y_train_seed = df_train_seed["fuel_efficiency_mpg"].to_numpy()
        X_val_seed = df_val_seed.drop(columns="fuel_efficiency_mpg").to_numpy()
        y_val_seed = df_val_seed["fuel_efficiency_mpg"].to_numpy()

        intercept_seed, weights_seed = train_linear_regression(X_train_seed, y_train_seed)
        y_pred_seed = intercept_seed + X_val_seed.dot(weights_seed)
        rmse_by_seed[seed_value] = rmse(y_val_seed, y_pred_seed)

    std_rmse = round(np.std(list(rmse_by_seed.values())), 3)
    rmse_by_seed, std_rmse
    return rmse_by_seed, std_rmse


@app.cell(hide_code=True)
def _(mo, rmse_by_seed, std_rmse):
    mo.md(f"""
    RMSE by seed: **{rmse_by_seed}**  \nStandard deviation: **{std_rmse}**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 6 - What is the RMSE on the test dataset?
    """)
    return


@app.cell
def _(df, np, prepare_validation_framework, rmse, train_linear_regression_reg):
    df_train_q6, df_val_q6, df_test_q6 = prepare_validation_framework(df, seed=9, fill_method="with_0")

    X_train_q6 = df_train_q6.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_train_q6 = df_train_q6["fuel_efficiency_mpg"].to_numpy()
    X_val_q6 = df_val_q6.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_val_q6 = df_val_q6["fuel_efficiency_mpg"].to_numpy()
    X_test_q6 = df_test_q6.drop(columns="fuel_efficiency_mpg").to_numpy()
    y_test_q6 = df_test_q6["fuel_efficiency_mpg"].to_numpy()

    X_train_val_q6 = np.concatenate([X_train_q6, X_val_q6])
    y_train_val_q6 = np.concatenate([y_train_q6, y_val_q6])

    intercept_q6, weights_q6 = train_linear_regression_reg(X_train_val_q6, y_train_val_q6, r=0.001)
    y_pred_test_q6 = intercept_q6 + X_test_q6.dot(weights_q6)
    rmse_test_q6 = round(rmse(y_test_q6, y_pred_test_q6), 3)
    rmse_test_q6
    return


if __name__ == "__main__":
    app.run()
