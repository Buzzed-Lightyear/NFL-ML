import pandas as pd

# --- 1. Load your CSV file ---
# Make sure the path to your CSV file is correct.
# If your CSV file is in the same directory as your Jupyter Notebook,
# you can just use the filename.
file_path = './nfl_data/2021/final-2021.csv'  # <--- IMPORTANT: REPLACE 'your_nfl_data.csv' WITH YOUR ACTUAL FILENAME

try:
    df = pd.read_csv(file_path)

    print("--- Successfully loaded the CSV file! ---")
    print(f"File loaded: {file_path}\n")

    # --- 2. Display the first few rows ---
    # This helps you see the column names and some sample data.
    print("--- First 5 rows of the dataset: ---")
    print(df.head())
    print("\n")

    # --- 3. Get basic information about the DataFrame ---
    # This shows data types of each column, number of non-null values, and memory usage.
    # Pay attention to Dtype (data type) and non-null counts.
    # Are all numerical columns showing as numbers (int64, float64)?
    # Are there any columns with missing values?
    print("--- DataFrame Info: ---")
    df.info()
    print("\n")

    # --- 4. Get descriptive statistics ---
    # This provides count, mean, std (standard deviation), min, max, and quartiles
    # for numerical columns. It's useful for spotting potential outliers or errors.
    print("--- Descriptive Statistics (for numerical columns): ---")
    print(df.describe())
    print("\n")

    # --- 5. Check for missing values in each column ---
    # This will show you the total number of missing (NaN) values per column.
    print("--- Missing Values per Column: ---")
    print(df.isnull().sum())
    print("\n")

    # --- 6. Check the shape of the DataFrame ---
    # This tells you (number_of_rows, number_of_columns).
    # The paper mentions "568 rows" for the statistics dataset before combining/processing [cite: 231]
    # and the final dataset for modeling had 568 observations[cite: 294].
    # Your row count might be similar if you used two full seasons.
    # The paper used 57 variables for the RFC model [cite: 323] before the target was added.
    # You mentioned having 2 fewer columns than the paper's RFC model,
    # so if the paper used 57 features + 1 target = 58 columns, you might have around 56.
    print("--- Shape of the DataFrame (rows, columns): ---")
    print(df.shape)
    print("\n")

    # --- 7. List all column names ---
    # This is a good final check to see all the features you have.
    print("--- Column Names: ---")
    print(list(df.columns))
    print("\n")

    # --- 8. Verify the target variable ---
    # Let's check the unique values in your 'team1_win' column.
    # It should ideally be 0s and 1s. The paper mentions ties were counted as "non-wins"[cite: 201].
    if 'team1_win' in df.columns:
        print("--- Unique values in the target variable 'team1_win': ---")
        print(df['team1_win'].value_counts(dropna=False)) # dropna=False will also show count of NaNs if any
    else:
        print("ERROR: Target variable 'team1_win' not found in the columns!")
        print("Please ensure your target column is named correctly.")

except FileNotFoundError:
    print(f"ERROR: The file '{file_path}' was not found.")
    print("Please make sure the file name and path are correct and the file is in the specified location.")
except Exception as e:
    print(f"An error occurred: {e}")