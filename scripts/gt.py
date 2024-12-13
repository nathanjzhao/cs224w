import pandas as pd

def print_unique_gt(df):
    # Print all unique values in the 'GT' column
    unique_gt = df['GT'].unique()
    print("Unique values in 'GT' column:", unique_gt)


def print_unique_extra(df):
    # Print all unique values in the 'extra' column
    unique_extra = df['extra'].unique()
    print("Unique values in 'extra' column:", unique_extra)

def main():
    # Load the CSV file
    df = pd.read_csv('data/GT.csv')

    # Filter rows where 'extra' column is not NaN
    filtered_df = df[df['extra'].notna()]

    # Print the full rows of the filtered DataFrame without truncation
    print(repr(filtered_df))

    print("*" * 20)

    # Call the new function to print unique 'GT' values
    print_unique_gt(df)

    print("*" * 20)

    # Call the new function to print unique 'extra' values
    print_unique_extra(df)

if __name__ == "__main__":
    main()