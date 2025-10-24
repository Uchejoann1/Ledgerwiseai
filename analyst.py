import os
import json
import boto3
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List
import instructor
from instructor import Mode

# --- 1. Define the Structured Output Schema (Pydantic Model) ---
# This defines the exact report structure the AI must return.
class BusinessAnalysisReport(BaseModel):
    """
    A comprehensive business analysis and advisory report.
    """
    profitability_analysis: str = Field(
        ...,
        description="Detailed 1-2 paragraph analysis of the business's Net Profit and Profit Margin (Net Profit / Revenue), explaining what these numbers mean for the business's health."
    )
    growth_and_future_projection: str = Field(
        ...,
        description="A 1-2 paragraph analysis of growth potential, comparing the profit margin to Nigerian industry benchmarks, followed by a 3-6 month financial projection based on current data."
    )
    business_efficiency_analysis: str = Field(
        ...,
        description="A 1-2 paragraph analysis of cost efficiency by analyzing the Cost-to-Revenue ratio (Total Costs / Revenue). Explain what this ratio means for profitability."
    )
    estimated_business_valuation: str = Field(
        ...,
        description="A *very high-level, theoretical* estimated valuation (e.g., using a 2x-3x SDE multiple on net profit). This *must* include a disclaimer that it is not a formal valuation and is for informational purposes only."
    )
    tax_compliance_overview: str = Field(
        ...,
        description="A high-level overview of potential Nigerian tax obligations (CIT, TET, VAT) based on their revenue and profit. This is *not* a calculation, but an advisory on their likely tax bracket."
    )
    loan_eligibility_assessment: str = Field(
        ...,
        description="A high-level assessment of loan eligibility based on profitability (Net Profit) and liquidity (Bank Balance), considering general Nigerian banking criteria (e.g., positive cash flow, profitability). This is not a guarantee of a loan."
    )
    actionable_advice: List[str] = Field(
        ...,
        description="A bulleted list of 2-3 specific, actionable recommendations for improvement, directly tied to the analysis."
    )

# --- 2. System Instruction and LLM Configuration ---
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """
You are an expert Nigerian Business Analyst, Financial Valuator, and Tax Advisor. Your goal is to analyze a small business's monthly financial data and provide a *detailed, 7-part, actionable report*.

**YOUR 7-PART TASK:**
You will be given the user's industry, monthly revenue, total monthly costs, and current bank balance. You MUST analyze this data and generate a report covering these 7 areas:

1.  **Profitability Analysis:** Calculate Net Profit (Revenue - Total Costs) and Profit Margin (Net Profit / Revenue). Analyze these figures in detail.
2.  **Growth & Future Projection:** Use your general knowledge to establish a *reasonable benchmark profit margin* for their specific industry *in Nigeria*. Compare their margin to this benchmark and provide a 3-6 month projection.
3.  **Business Efficiency Analysis:** Analyze their Cost-to-Revenue ratio (Total Costs / Revenue). Discuss their efficiency and how it impacts profit potential.
4.  **Estimated Business Valuation:** Calculate a *theoretical* valuation. Use a simple SDE (Seller's Discretionary Earnings, approximated as Net Profit) multiple (e.g., 2.0x - 3.0x annual Net Profit). **You MUST include a disclaimer** stating: "This is a theoretical estimate for informational purposes only and not a formal valuation."
5.  **Tax Compliance Overview:** Based on their *Total Revenue* (annualized = Monthly Revenue * 12), inform them of their likely Nigerian tax obligations.
    * If Annualized Revenue <= 25M NGN: "Small Company" (0% CIT, VAT exempt).
    * If 25M < Annualized Revenue <= 100M NGN: "Medium Company" (20% CIT, 3% TET, VAT applicable).
    * If Annualized Revenue > 100M NGN: "Large Company" (30% CIT, 3% TET, VAT applicable).
    Advise them to consult a professional. Do *not* calculate the exact tax amount.
6.  **Loan Eligibility Assessment:** Based on their Net Profit (cash flow) and Bank Balance (liquidity), provide a high-level assessment of their eligibility for a business loan in Nigeria. Mention that banks look for profitability and stable cash flow. **This is not a guarantee of a loan.**
7.  **Actionable Advice:** Provide 2-3 specific, bulleted recommendations for improvement.

You MUST return your entire response as a valid JSON object matching the 'BusinessAnalysisReport' schema. Do NOT include any text outside the JSON.
"""

def get_business_analysis(user_data: dict) -> BusinessAnalysisReport:
    """
    Calls the Llama 3 70B model to generate a structured business analysis.
    """
    try:
        # 1. Initialize Bedrock Boto3 Client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=AWS_REGION
        )

        # 2. Patch the client with instructor
        client = instructor.from_bedrock(
            bedrock_client,
            mode=Mode.BEDROCK_JSON 
        )
        
        # Format the user's data into a string for the LLM
        data_string = f"""
        --- USER FINANCIAL DATA (1 Month) ---
        Industry: {user_data.get('industry')}
        Monthly Revenue: {user_data.get('revenue'):,.2f} NGN
        Total Monthly Costs: {user_data.get('total_costs'):,.2f} NGN
        Current Bank Account Balance: {user_data.get('bank_balance'):,.2f} NGN
        ---
        Calculated Net Profit/Loss: {user_data.get('net_profit'):,.2f} NGN
        ---
        """
        
        # Llama 3 works best by combining the system prompt and user query
        full_query = f"{SYSTEM_PROMPT}\n\nAnalyze the following data:\n{data_string}"

        print(f"\n-> Analyzing data with {MODEL_ID}...")
        # 3. Create the structured completion request
        report_object = client.messages.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": full_query} # Combined prompt
            ],
            response_model=BusinessAnalysisReport,
            max_tokens=2048, # Max limit for Llama 3 70B on Bedrock
            temperature=0.1 # Low temp for factual analysis
        )

        return report_object

    except Exception as e:
        print(f"\nAn API/Connection error occurred: {e}")
        # Return a fallback object on severe failure
        return BusinessAnalysisReport(
            profitability_analysis="Error: Could not generate analysis. API connection failed.",
            growth_and_future_projection="N/A",
            business_efficiency_analysis="N/A",
            estimated_business_valuation="N/A",
            tax_compliance_overview="N/A. Check AWS credentials and model access for Llama 3 70B.",
            loan_eligibility_assessment="N/A",
            actionable_advice=["API connection failed."]
        )

def get_numeric_input(prompt: str) -> float:
    """Helper function to safely get float input from the user."""
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['quit', 'exit', 'q']:
            raise SystemExit("Exiting analyst tool. Goodbye!")
        
        try:
            # Remove commas if user enters them (e.g., 1,000,000)
            value = float(user_input.replace(',', ''))
            if value < 0:
                print("Value cannot be negative. Please try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value (e.g., 500000).")

# --- 4. Interactive Execution Loop ---
if __name__ == "__main__":
    print("\n--- 🇳🇬 AI Business Analyst (Nigeria) ---")
    print("This tool collects your monthly financial data to provide a 7-part analysis.")
    print("Type 'quit' or 'exit' at any prompt to end the session.\n")
    
    while True:
        try:
            # --- 1. Collect Data ---
            print("-" * 50)
            industry = input("What is your business industry (e.g., 'Retail', 'Restaurant', 'Logistics')?\n> ").strip()
            if industry.lower() in ['quit', 'exit', 'q']:
                print("\nExiting analyst tool. Goodbye!")
                break
            if not industry:
                print("Industry is required to provide a benchmark. Please try again.")
                continue

            revenue = get_numeric_input("Enter your total Monthly Revenue (NGN):\n> ")
            total_costs = get_numeric_input("Enter your total Monthly Costs (Fixed + Variable) (NGN):\n> ")
            bank_balance = get_numeric_input("Enter your Current Business Bank Account Balance (NGN):\n> ")
            
            # --- 2. Pre-Calculation ---
            net_profit = revenue - total_costs
            
            print("\n--- Your Data Summary ---")
            print(f"  Monthly Revenue:    ₦{revenue:,.2f}")
            print(f"  Total Monthly Cost: ₦{total_costs:,.2f}")
            print(f"  Net Profit/Loss:    ₦{net_profit:,.2f}")
            print(f"  Bank Balance:       ₦{bank_balance:,.2f}")
            
            if revenue == 0:
                print("\nCannot calculate profit margin or provide analysis with zero revenue.")
                continue

            # --- 3. Get AI Analysis ---
            user_data_payload = {
                "industry": industry,
                "revenue": revenue,
                "total_costs": total_costs,
                "bank_balance": bank_balance,
                "net_profit": net_profit
            }
            
            report = get_business_analysis(user_data_payload)
            
            # --- 4. Display Report (UPDATED for new fields) ---
            print("\n" + "="*50)
            print("| AI BUSINESS ANALYSIS REPORT |")
            print("="*50)

            print("\n--- 1. PROFITABILITY ANALYSIS ---")
            print(report.profitability_analysis)

            print("\n--- 2. GROWTH & FUTURE PROJECTION ---")
            print(report.growth_and_future_projection)

            print("\n--- 3. BUSINESS EFFICIENCY ANALYSIS ---")
            print(report.business_efficiency_analysis)

            print("\n--- 4. ESTIMATED BUSINESS VALUATION ---")
            print(report.estimated_business_valuation)

            print("\n--- 5. TAX COMPLIANCE OVERVIEW ---")
            print(report.tax_compliance_overview)

            print("\n--- 6. LOAN ELIGIBILITY ASSESSMENT ---")
            print(report.loan_eligibility_assessment)

            print("\n--- 7. ACTIONABLE ADVICE ---")
            for item in report.actionable_advice:
                print(f"  • {item}")
            print("="*50)

        except SystemExit:
            break
        except Exception as e:
            print(f"\nAn unexpected runtime error occurred: {e}. Restarting loop.")
            
        # --- 5. Continuation Prompt ---
        continue_prompt = input("\nPress Enter to run a new analysis, or type 'no' to exit:\n> ").strip().lower()
        if continue_prompt in ['no', 'n', 'quit', 'exit']:
            print("\nExiting analyst tool. Goodbye!")
            break