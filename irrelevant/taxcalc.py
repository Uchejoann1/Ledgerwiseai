import os
import json
import boto3
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
import instructor
from instructor import Mode

# --- 1. Define the Structured Output Schema (Pydantic Model) ---
# This defines the exact structure and fields the AI must return.
class TaxCalculationResult(BaseModel):
    """
    Structured result containing the tax calculation, business advice, and compliance assessment.
    """
    taxable_profit: float = Field(
        ...,
        description="The final calculated taxable profit (Revenue - Allowable Deductions)."
    )
    cit_rate_applied: float = Field(
        ...,
        description="The Corporate Income Tax (CIT) rate applied (e.g., 20.0 or 30.0), based on company turnover."
    )
    cit_liability: float = Field(
        ...,
        description="The total calculated Corporate Income Tax (CIT) due."
    )
    education_tax_liability: float = Field(
        ...,
        description="The calculated Tertiary Education Tax (TET) due at 3% of the assessable profit."
    )
    total_tax_due: float = Field(
        ...,
        description="The sum of CIT and Education Tax liabilities."
    )
    compliance_status: Literal["COMPLIANT", "NON_COMPLIANT", "UNKNOWN"] = Field(
        ...,
        description="Assessment of whether the company is compliant based on the calculated tax (COMPLIANT if tax is > 0, NON_COMPLIANT if tax is < 0 or if crucial data is missing, UNKNOWN if data is insufficient)."
    )
    compliance_recommendation: str = Field(
        ...,
        description="Actionable business and tax advice based on the company's financial results and compliance status, specific to Nigerian regulations."
    )

# --- 2. System Instruction and LLM Configuration ---
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """
You are a highly specialized Nigerian Corporate Tax and Business Advisory AI. Your sole function is to calculate tax liabilities and provide expert advice based on provided financial data and FIRS regulations.

TAX RULES (Based on Nigerian Finance Act 2019/2020):
1. Small Company (Turnover <= ₦25,000,000): CIT Rate is 0%.
2. Medium Company (₦25,000,001 < Turnover <= ₦100,000,000): CIT Rate is 20%.
3. Large Company (Turnover > ₦100,000,000): CIT Rate is 30%.
4. Tertiary Education Tax (TET): 3% of Assessable Profit (same as taxable profit).

INSTRUCTIONS:
1. Identify Total Revenue, Cost of Sales, and Operating Expenses from the provided raw data.
2. Calculate Assessable/Taxable Profit: Total Revenue - Cost of Sales - Operating Expenses.
3. Determine the CIT Rate based on the 'Total Revenue' figure.
4. Calculate CIT Liability: Taxable Profit * CIT Rate.
5. Calculate TET Liability: Taxable Profit * 0.03 (3%).
6. Calculate Total Tax Due: CIT Liability + TET Liability.
7. Provide comprehensive compliance advice and business recommendations in the 'compliance_recommendation' field.

You MUST return your response as a valid JSON object matching the provided schema.
"""

def load_financial_data(filepath: str) -> tuple[pd.DataFrame | None, float | None]:
    """
    Reads data from a CSV or XLSX file and attempts to extract Total Revenue.
    """
    try:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.lower().endswith('.xlsx') or filepath.lower().endswith('.xls'):
            # Requires openpyxl: pip install openpyxl
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")
        
        # Enhanced extraction logic: prioritize finding 'Revenue'
        # Convert all columns to string and search for 'revenue' metric label
        metric_col_names = ['metric', 'item', 'description', 'particulars', 'details']
        amount_col_names = ['amount', 'value', 'ngn', 'total', 'cost']
        
        metric_col = next((col for col in df.columns if any(name in str(col).lower() for name in metric_col_names)), None)
        amount_col = next((col for col in df.columns if any(name in str(col).lower() for name in amount_col_names)), None)

        if metric_col and amount_col:
            # Try to find the Total Revenue row
            revenue_row = df[df[metric_col].astype(str).str.contains('revenue|sales', case=False, na=False)]
            if not revenue_row.empty:
                total_revenue = revenue_row[amount_col].iloc[0]
            else:
                raise KeyError("Could not identify the Metric or Amount columns in the file. Please ensure one column is labeled like 'Metric' or 'Description' and another like 'Amount' or 'NGN'.")
        
        return df, float(total_revenue)
    
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {filepath}")
        return None, None
    except KeyError as e:
        print(f"ERROR: Column identification error. {e}")
        return None, None
    except ValueError as e:
        print(f"ERROR: Data or format error: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None, None


def calculate_tax_and_assess(business_size: str, filepath: str) -> TaxCalculationResult:
    """
    Loads financial data from file, sends it to the LLM for calculation, and returns the structured result.
    """
    # Load data from the CSV/XLSX file
    financial_data, total_revenue = load_financial_data(filepath)

    if financial_data is None:
        # Return a structured error if data loading failed
        return TaxCalculationResult(
            taxable_profit=0.0, cit_rate_applied=0.0, cit_liability=0.0,
            education_tax_liability=0.0, total_tax_due=0.0, compliance_status="UNKNOWN",
            compliance_recommendation="Data loading failed. Check file path, existence, and column names."
        )

    # Convert DataFrame to string for the LLM to process
    financial_data_str = financial_data.to_string(index=False)

    # 1. Initialize Bedrock Boto3 Client
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=AWS_REGION
        )

        # 2. Patch the client with instructor (using BEDROCK_JSON for Llama 3)
        client = instructor.from_bedrock(
            bedrock_client,
            mode=Mode.BEDROCK_JSON 
        )
    except Exception as e:
        # Fallback for AWS initialization failure
        return TaxCalculationResult(
            taxable_profit=0.0, cit_rate_applied=0.0, cit_liability=0.0,
            education_tax_liability=0.0, total_tax_due=0.0, compliance_status="UNKNOWN",
            compliance_recommendation=f"AWS setup or connectivity failed. Check credentials and Bedrock access in region {AWS_REGION}."
        )

    # Combine data and prompt
    user_query = f"""
    Please calculate the tax liabilities and compliance status for a Nigerian company of '{business_size}' size using the following financial statement data (Amounts in NGN). The Total Revenue found in the file is approximately {total_revenue:,.2f} NGN.

    --- FINANCIAL DATA ---
    {financial_data_str}
    ---

    Follow all the Nigerian tax rules provided in the system instruction exactly and return ONLY the JSON object.
    """

    print(f"\n-> Calculating tax for {business_size} company using data from {filepath} (Llama 3 70B)...")
    try:
        # 3. Create the structured completion request
        result_object = client.messages.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_query}"}
            ],
            response_model=TaxCalculationResult,
            max_tokens=2048,
            temperature=0.0
        )
        return result_object

    except Exception as e:
        print(f"An API/Calculation error occurred: {e}")
        # Return a fallback object on API failure
        return TaxCalculationResult(
            taxable_profit=0.0, cit_rate_applied=0.0, cit_liability=0.0,
            education_tax_liability=0.0, total_tax_due=0.0, compliance_status="UNKNOWN",
            compliance_recommendation="API request failed. Check model permissions and AWS access for Llama 3 70B."
        )


if __name__ == "__main__":
    
    print("--- 🇳🇬 Nigerian Corporate Tax & Compliance Calculator (File Uploader) ---")
    print("Processes CSV or Excel data using AI to calculate tax liabilities.")
    
    while True:
        print("-" * 60)
        
        # 1. Get File Path
        filepath = input("Enter the path to your CSV or Excel (.xlsx) file, or 'exit': ").strip()
        if filepath.lower() in ['exit', 'quit', 'q']:
            print("Exiting calculator. Goodbye!")
            break
            
        # 2. Get Business Size
        business_size = input("Enter business size for context (MEDIUM or LARGE): ").strip().upper()
        if business_size not in ["MEDIUM", "LARGE"]:
            print("Invalid business size. Please enter 'MEDIUM' or 'LARGE'.")
            continue

        # Calculate and assess tax
        result = calculate_tax_and_assess(business_size, filepath)

        # --- Display Final Output ---
        print("\n" + "=" * 60)
        print(f"| TAX ASSESSMENT FOR {business_size} COMPANY |")
        print("=" * 60)
        
        # Format currency with commas
        def format_currency(amount):
            return f"₦{amount:,.2f}"

        # If a system error occurred during data loading or API call, print the error message cleanly
        if result.compliance_status == "UNKNOWN":
            print(f"SYSTEM ERROR: {result.compliance_recommendation}")
        else:
            print(f"  > Taxable Profit:        {format_currency(result.taxable_profit)}")
            print(f"  > CIT Rate Applied:      {result.cit_rate_applied:.1f}%")
            print(f"  > CIT Liability:         {format_currency(result.cit_liability)}")
            print(f"  > TET Liability (3%):    {format_currency(result.education_tax_liability)}")
            print("-" * 60)
            print(f"  > TOTAL TAX DUE:         {format_currency(result.total_tax_due)}")
            print("=" * 60)

            # Compliance Assessment
            print(f"\nCOMPLIANCE STATUS: {result.compliance_status}")
            print(f"\nRECOMMENDATION & ADVICE:\n{result.compliance_recommendation}")
            
        print("\n" + "=" * 60)
        
        # Prompt for continuation
        continue_prompt = input("Press Enter to run another calculation, or type 'no' to exit: ").strip().lower()
        if continue_prompt in ['no', 'n', 'quit', 'exit']:
            print("Exiting calculator. Goodbye!")
            break
