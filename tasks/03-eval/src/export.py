import pandas as pd

df: pd.DataFrame = pd.read_csv("data/thesis.csv")
df = df[["mean", "std", "95%", "max"]]
print(
    df.to_string(
        formatters={
            "mean": "{:.2f}".format,
            "std": "{:.2f}".format,
            "95%": "{:.2f}".format,
            "max": "{:.2f}".format,
        }
    )
)
print(f'mean: {df["mean"].mean():.2f}')
print(f'std: {df["std"].mean():.2f}')
print(f'95%: {df["95%"].mean():.2f}')
print(f'max: {df["max"].mean():.2f}')
