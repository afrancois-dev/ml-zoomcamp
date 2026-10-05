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
def _(np, pd, seed):
    def prepare_validation_framework(df: pd.DataFrame, int: seed, type: "with_0" | "with_mean"):
        if type == "with_0":
            df["horsepower"].fillna(0)
        else:
            df["horsepower"].fillna(df["horsepower"].mean())
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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### with 0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### with mean
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
