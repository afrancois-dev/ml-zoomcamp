import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():

    return


@app.cell
def _():
    import pandas as pd

    pd.__version__
    return (pd,)


@app.cell
def _(pd):
    pd.read_csv("car_fuel_efficiency_2026.csv")
    return


if __name__ == "__main__":
    app.run()
