import pandas as pd

print("Generating sample Excel file for testing...")

# 1. Define the sample financial data for a "Large Company"
# The column names 'Metric' and 'Amount_NGN' match what the calculator script looks for.
data = {
    'Metric': [
        'Total Revenue', 
        'Cost of Sales', 
        'Operating Expenses', 
        'Other Income (Non-taxable)', 
        'Depreciation (Non-allowable)',
        'Tax'
    ],
    'Amount_NGN': [
        145_000_000,  # Turnover > 100M (Large Company)
        60_000_000,
        35_000_000,
        2_000_000,
        5_000_000,
        4_500_000
    ]
}

# 2. Create a Pandas DataFrame
df = pd.DataFrame(data)

# 3. Define the output filename
filename = 'large_company_data.xlsx'

try:
    # 4. Save the DataFrame to an Excel file
    # We use index=False so the row numbers (0, 1, 2...) aren't saved as a column
    # 'openpyxl' is the engine pandas uses for .xlsx files
    df.to_excel(filename, index=False, engine='openpyxl')
    
    print(f"\nSuccessfully generated '{filename}'.")
    print("You can now use this file as input for your tax calculator.")

except ImportError:
    print("\nERROR: Could not generate file.")
    print("Please install the 'openpyxl' library to write Excel files:")
    print("pip install openpyxl")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

