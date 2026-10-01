import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1. Pandas version
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    pd.__version__
    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("car_fuel_efficiency_2026.csv")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2. Records count
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q3. Fuel types
    """)
    return


@app.cell
def _(df):
    df["fuel_type"].unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4. Missing values
    """)
    return


@app.cell
def _(df):
    df.isna()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q5. Max fuel efficiency
    """)
    return


@app.cell
def _(df):
    df[df["origin"] == "Asia"]["fuel_efficiency_mpg"].max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q6. Median value of horsepower
    """)
    return


@app.cell
def _(df):
    median_before = df["horsepower"].median()
    most_frequent = df["horsepower"].mode()[0]
    df["horsepower"] = df["horsepower"].fillna(most_frequent)
    median_after = df["horsepower"].median()
    median_before, most_frequent, median_after
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q7. Sum of weights
    """)
    return


@app.cell
def _(df, np):
    X = df[df["origin"] == "Asia"][["vehicle_weight", "model_year"]].head(7).to_numpy()
    XTX = X.T @ X
    XTX_inverse = np.linalg.inv(XTX)
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    w = XTX_inverse @ X.T @ y
    w.sum()
    return


if __name__ == "__main__":
    app.run()
