"""
CodeAlpha - Task 2: Exploratory Data Analysis (EDA)
Analyzes book data collected in Task 1 (books_data.csv) to answer
meaningful questions, spot patterns, and flag
data quality issues.


pip install pandas numpy scipy matplotlib seaborn

"""

import pandas as pd
import numpy as np
from scipy import stats

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

def load_data(path: str = "Data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns\n")
    return df


def explore_structure(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("1. DATA STRUCTURE")
    print("=" * 60)
    print("\nColumn types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary statistics (numeric columns):")
    print(df.describe())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())


def ask_questions(df: pd.DataFrame) -> None:
    """Meaningful questions to ask before diving into analysis."""
    print("\n" + "=" * 60)
    print("2. MEANINGFUL QUESTIONS ABOUT THE DATA")
    print("=" * 60)

    questions = [
        "Do higher-rated books tend to cost more?",
        "What share of books are currently out of stock?",
        "Is price distributed normally, or are there outliers/skew?",
        "Does stock status differ meaningfully by rating tier?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"  Q{i}: {q}")


def find_trends_and_patterns(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. TRENDS, PATTERNS AND ANOMALIES")
    print("=" * 60)

    print("\nAverage price by rating:")
    print(df.groupby("rating")["price_gbp"].mean().round(2))

    print("\nBook count by rating:")
    print(df["rating"].value_counts().sort_index())

    print("\nIn-stock vs out-of-stock counts:")
    print(df["in_stock"].value_counts())

    # Simple outlier detection using IQR
    q1 = df["price_gbp"].quantile(0.25)
    q3 = df["price_gbp"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df[(df["price_gbp"] < lower) | (df["price_gbp"] > upper)]
    print(f"\nPrice outliers (IQR method): {len(outliers)} books "
          f"outside [{lower:.2f}, {upper:.2f}]")


def test_hypothesis(df: pd.DataFrame) -> None:
    """
    Hypothesis: Books with a 5-star rating are priced differently
    from books with a 1-star rating.
    H0: no difference in mean price between 1-star and 5-star books.
    H1: there is a difference.
    """
    print("\n" + "=" * 60)
    print("4. HYPOTHESIS TEST")
    print("=" * 60)
    print("H0: Mean price is the same for 1-star and 5-star books")
    print("H1: Mean price differs between 1-star and 5-star books\n")

    one_star = df[df["rating"] == 1]["price_gbp"]
    five_star = df[df["rating"] == 5]["price_gbp"]

    if len(one_star) < 2 or len(five_star) < 2:
        print("Not enough data in one of the groups to run the test.")
        return

    t_stat, p_value = stats.ttest_ind(one_star, five_star, equal_var=False)
    print(f"1-star books: n={len(one_star)}, mean price=£{one_star.mean():.2f}")
    print(f"5-star books: n={len(five_star)}, mean price=£{five_star.mean():.2f}")
    print(f"\nt-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"p < {alpha} -> Reject H0. There IS a statistically "
              f"significant price difference by rating.")
    else:
        print(f"p >= {alpha} -> Fail to reject H0. No statistically "
              f"significant price difference found.")


def flag_data_issues(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("5. DATA QUALITY ISSUES")
    print("=" * 60)

    issues = []
    if df.isnull().sum().sum() > 0:
        issues.append("Missing values present in one or more columns.")
    if df.duplicated().sum() > 0:
        issues.append(f"{df.duplicated().sum()} duplicate rows found.")
    if (df["price_gbp"] <= 0).any():
        issues.append("Some books have a zero or negative price.")
    if df["rating"].isnull().any():
        issues.append("Some books have no rating recorded.")

    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("  No major data quality issues detected.")


if __name__ == "__main__":
    df = load_data()
    explore_structure(df)
    ask_questions(df)
    find_trends_and_patterns(df)
    test_hypothesis(df)
    flag_data_issues(df)