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
    Structured result containing Profit Tax (CIT/TET) and VAT calculations,
    plus compliance status and business advice.
    """
    # --- PROFIT TAX SECTION ---
    taxable_profit: float = Field(
        ...,
        description="The final calculated taxable profit (Revenue - Allowable Deductions)."
    )
    cit_rate_applied: float = Field(
        ...,
        description="The Corporate Income Tax (CIT) rate applied (e.g., 20.0 or 30.0)."
    )
    cit_liability: float = Field(
        ...,
        description="The total calculated Corporate Income Tax (CIT) due."
    )
    education_tax_liability: float = Field(
        ...,
        description="The calculated Tertiary Education Tax (TET) due at 3% of the assessable profit."
    )
    total_profit_tax_due: float = Field(
        ...,
        description="The sum of CIT and Education Tax liabilities."
    )
    profit_tax_paid_by_user: float = Field(
        ...,
        description="The amount of *profit tax* (CIT/TET) the user has already paid."
    )
    profit_tax_payment_status_amount: float = Field(
        ...,
        description="The difference between profit tax paid and tax due (Paid - Due). Negative means underpaid, positive means overpaid."
    )
    
    # --- NEW VAT SECTION ---
    vat_output_collected: float = Field(
        ...,
        description="The amount of Output VAT (VAT on Sales) found in the user's data."
    )
    vat_input_paid: float = Field(
        ...,
        description="The amount of Input VAT (VAT on Purchases/Expenses) found in the user's data."
    )
    vat_remittable_due: float = Field(
        ...,
        description="The final VAT liability to be remitted to FIRS (Output VAT - Input VAT)."
    )

    # --- ADVISORY SECTION ---
    compliance_status: Literal[
        "COMPLIANT (Paid in Full)", 
        "NON_COMPLIANT (Underpaid)", 
        "OVERPAID (Refund Due)",
        "UNKNOWN (Check Data)"
    ] = Field(
        ...,
        description="Assessment of *Profit Tax* compliance based on the calculated tax vs. tax paid."
    )
    compliance_recommendation: str = Field(
        ...,
        description="Actionable advice *only* related to tax compliance (both Profit Tax and VAT), payment deadlines, and addressing payment status."
    )
    business_growth_advice: str = Field(
        ...,
        description="Actionable advice on how to improve or grow the business, based on analyzing the provided financial data (e.g., 'Your Cost of Sales is high...')."
    )


# --- 2. System Instruction and LLM Configuration ---
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# --- UPDATED SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a highly specialized Nigerian Corporate Tax and Business Advisory AI. Your function is to:
1.  Calculate exact Profit Tax (CIT & TET) liabilities.
2.  Calculate exact VAT Remittable liability.
3.  Audit payments and provide actionable business growth advice.

--- TAX RULES (Based on Nigerian Finance Act) ---
A. PROFIT TAX (CIT & TET):
1.  Small Company (Turnover <= ₦25,000,000): CIT Rate is 0%.
2.  Medium Company (₦25,000,001 < Turnover <= ₦100,000,000): CIT Rate is 20%.
3.  Large Company (Turnover > ₦100,000,000): CIT Rate is 30%.
4.  Tertiary Education Tax (TET): 3% of Assessable Profit (same as taxable profit).

B. VALUE ADDED TAX (VAT):
1.  VAT Rate: 7.5%.
2.  VAT Threshold: Applies to companies with turnover > ₦25,000,000.
3.  VAT Remittable = Output VAT (VAT on Sales) - Input VAT (VAT on Purchases).
4.  Input VAT can only be claimed on goods purchased for resale or used directly in production. You can assume Input VAT provided is claimable.

--- INSTRUCTIONS ---
1.  Identify Total Revenue, Cost of Sales, Operating Expenses, Output VAT, Input VAT, and Profit Tax Paid from the user's data.
2.  Calculate Assessable/Taxable Profit: Total Revenue - Cost of Sales - Operating Expenses.
3.  Determine the CIT Rate based on 'Total Revenue'.
4.  Calculate CIT Liability: Taxable Profit * CIT Rate.
5.  Calculate TET Liability: Taxable Profit * 0.03 (3%).
6.  Calculate Total Profit Tax Due: CIT Liability + TET Liability.
7.  Calculate Profit Tax Payment Status Amount: Profit Tax Paid By User - Total Profit Tax Due.
8.  Calculate VAT Remittable Due: Output VAT - Input VAT. (This should be 0 if Total Revenue is <= ₦25M).
9.  Set 'compliance_status' based *only* on the 'Profit Tax Payment Status Amount'.
10. Provide comprehensive *tax compliance* advice in 'compliance_recommendation', covering *both* profit tax and VAT liabilities.
11. Provide actionable *business growth* advice in 'business_growth_advice'.

You MUST return your response as a valid JSON object matching the provided schema.
"""

def load_financial_data(filepath: str) -> tuple[pd.DataFrame | None, float | None, float, float, float]:
    """
    Reads data from a CSV or XLSX file and attempts to extract key financial metrics.
    Returns: (DataFrame, TotalRevenue, ProfitTaxPaid, OutputVAT, InputVAT)
    """
    try:
        if filepath.lower().endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.lower().endswith('.xlsx') or filepath.lower().endswith('.xls'):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")
        
        # Enhanced extraction logic
        metric_col_names = ['metric', 'item', 'description', 'particulars', 'details']
        amount_col_names = ['amount', 'value', 'ngn', 'total', 'cost']
        
        metric_col = next((col for col in df.columns if any(name in str(col).lower() for name in metric_col_names)), None)
        amount_col = next((col for col in df.columns if any(name in str(col).lower() for name in amount_col_names)), None)

        if not (metric_col and amount_col):
            raise KeyError("Could not identify the Metric or Amount columns in the file. Please ensure one column is labeled like 'Metric' or 'Description' and another like 'Amount' or 'NGN'.")

        # Helper to find a value, defaulting to 0.0
        def find_value(keywords: list[str]) -> float:
            for keyword in keywords:
                row = df[df[metric_col].astype(str).str.contains(keyword, case=False, na=False)]
                if not row.empty:
                    return float(row[amount_col].iloc[0])
            return 0.0

        # Find Total Revenue (Mandatory)
        total_revenue = find_value(['total revenue', 'revenue', 'sales'])
        if total_revenue == 0.0:
            raise ValueError("Could not locate a row labeled 'Revenue' or 'Sales' in the data.")

        # Find other values (default to 0.0 if not found)
        profit_tax_paid = find_value(['profit tax paid', 'cit paid', 'tax paid'])
        output_vat = find_value(['output vat', 'vat collected', 'vat on sales'])
        input_vat = find_value(['input vat', 'vat paid on inputs', 'vat on purchases'])
        
        return df, total_revenue, profit_tax_paid, output_vat, input_vat
    
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {filepath}")
        return None, None, 0.0, 0.0, 0.0
    except KeyError as e:
        print(f"ERROR: Column identification error. {e}")
        return None, None, 0.0, 0.0, 0.0
    except ValueError as e:
        print(f"ERROR: Data or format error: {e}")
        return None, None, 0.0, 0.0, 0.0
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None, None, 0.0, 0.0, 0.0


def get_fallback_response(recommendation: str, profit_tax_paid: float = 0.0) -> TaxCalculationResult:
    """Helper function to create structured error responses."""
    return TaxCalculationResult(
        taxable_profit=0.0,
        cit_rate_applied=0.0,
        cit_liability=0.0,
        education_tax_liability=0.0,
        total_profit_tax_due=0.0,
        profit_tax_paid_by_user=profit_tax_paid,
        profit_tax_payment_status_amount=0.0,
        vat_output_collected=0.0,
        vat_input_paid=0.0,
        vat_remittable_due=0.0,
        compliance_status="UNKNOWN (Check Data)",
        compliance_recommendation=recommendation,
        business_growth_advice="N/A. Cannot provide business advice due to error."
    )


def calculate_tax_and_assess(business_size: str, filepath: str) -> TaxCalculationResult:
    """
    Loads financial data from file, sends it to the LLM for calculation, and returns the structured result.
    """
    # Load data from the CSV/XLSX file
    financial_data, total_revenue, profit_tax_paid, output_vat, input_vat = load_financial_data(filepath)

    if financial_data is None:
        # Return a structured error if data loading failed
        return get_fallback_response(
            "Data loading failed. Check file path, existence, and column names."
        )

    # Convert DataFrame to string for the LLM to process
    financial_data_str = financial_data.to_string(index=False)

    # 1. Initialize Bedrock Boto3 Client
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=AWS_REGION
        )
        client = instructor.from_bedrock(bedrock_client, mode=Mode.BEDROCK_JSON)
    except Exception as e:
        # Fallback for AWS initialization failure
        return get_fallback_response(
            f"AWS setup or connectivity failed. Check credentials and Bedrock access in region {AWS_REGION}."
        )

    # Combine data and prompt
    user_query = f"""
    Please calculate the tax liabilities, audit compliance, and provide business advice for a Nigerian company of '{business_size}' size using the following financial statement data (Amounts in NGN).

    --- FINANCIAL DATA (RAW) ---
    {financial_data_str}
    ---
    
    --- KEY EXTRACTED VALUES ---
    Total Revenue: {total_revenue:,.2f} NGN
    Profit Tax Paid by User: {profit_tax_paid:,.2f} NGN
    Output VAT (VAT on Sales): {output_vat:,.2f} NGN
    Input VAT (VAT on Purchases): {input_vat:,.2f} NGN
    ---

    Follow all the Nigerian tax rules and advisory instructions provided in the system prompt exactly and return ONLY the JSON object.
    """

    print(f"\n-> Calculating tax & generating advice for {business_size} company (Llama 3 70B)...")
    try:
        # 3. Create the structured completion request
        result_object = client.messages.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_query}"}
            ],
            response_model=TaxCalculationResult,
            max_tokens=2048,
            temperature=0.1 
        )
        return result_object

    except Exception as e:
        print(f"An API/Calculation error occurred: {e}")
        # Return a fallback object on API failure
        return get_fallback_response(
            "API request failed. Check model permissions and AWS access for Llama 3 70B.",
            profit_tax_paid=profit_tax_paid
        )


if __name__ == "__main__":
    
    print("--- 🇳🇬 Nigerian Corporate Tax & Business Advisor (File Uploader) ---")
    print("Processes CSV or Excel data using AI to calculate tax liabilities and give growth advice.")
    
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
        print(f"| TAX & BUSINESS ASSESSMENT FOR {business_size} COMPANY |")
        print("=" * 60)
        
        # Format currency with commas
        def format_currency(amount):
            return f"₦{amount:,.2f}"

        # If a system error occurred, print the error message cleanly
        if "failed" in result.compliance_recommendation.lower() or "check data" in result.compliance_status.lower():
            print(f"SYSTEM ERROR: {result.compliance_recommendation}")
            print(f"BUSINESS ADVICE: {result.business_growth_advice}")
        else:
            print("--- PROFIT TAX CALCULATION (Annual) ---")
            print(f"  > Taxable Profit:        {format_currency(result.taxable_profit)}")
            print(f"  > CIT Rate Applied:      {result.cit_rate_applied:.1f}%")
            print(f"  > CIT Liability:         {format_currency(result.cit_liability)}")
            print(f"  > TET Liability (3%):    {format_currency(result.education_tax_liability)}")
            print("-" * 60)
            print(f"  > TOTAL PROFIT TAX DUE:  {format_currency(result.total_profit_tax_due)}")
            print(f"  > PROFIT TAX PAID:       {format_currency(result.profit_tax_paid_by_user)}")
            print("-" * 60)
            
            # Display Underpayment or Overpayment
            if result.profit_tax_payment_status_amount < 0:
                print(f"  > PAYMENT STATUS:        {format_currency(result.profit_tax_payment_status_amount)} (Underpaid)")
            elif result.profit_tax_payment_status_amount > 0:
                print(f"  > PAYMENT STATUS:        {format_currency(result.profit_tax_payment_status_amount)} (Overpaid)")
            else:
                 print(f"  > PAYMENT STATUS:        {format_currency(result.profit_tax_payment_status_amount)} (Paid in Full)")
            
            print("=" * 60)

            # --- NEW VAT SECTION ---
            print(f"\n--- VAT CALCULATION (Monthly) ---")
            print(f"  > Output VAT (On Sales): {format_currency(result.vat_output_collected)}")
            print(f"  > Input VAT (On Purchases):{format_currency(result.vat_input_paid)}")
            print("-" * 60)
            print(f"  > VAT REMITTABLE TO FIRS:{format_currency(result.vat_remittable_due)}")
            print("=" * 60)

            # Compliance Assessment
            print(f"\n--- TAX COMPLIANCE ---")
            print(f"PROFIT TAX STATUS: {result.compliance_status}")
            print(f"\nRECOMMENDATION (Tax):\n{result.compliance_recommendation}")
            
            # --- NEW BUSINESS ADVICE SECTION ---
            print("\n" + "=" * 60)
            print(f"\n--- BUSINESS GROWTH ADVICE ---")
            print(f"{result.business_growth_advice}")
            
        print("\n" + "=" * 60)
        
        # Prompt for continuation
        continue_prompt = input("Press Enter to run another calculation, or type 'no' to exit: ").strip().lower()
        if continue_prompt in ['no', 'n', 'quit', 'exit']:
            print("Exiting calculator. Goodbye!")
            break