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
    A comprehensive business analysis report based on one month of financial data.
    """
    profit_margin_percentage: float = Field(
        ...,
        description="The user's net profit margin as a percentage (Net Profit / Revenue * 100)."
    )
    performance_summary: str = Field(
        ...,
        description="A 1-2 sentence summary of the business's performance for the month."
    )
    industry_benchmark: str = Field(
        ...,
        description="A comparison of the user's profit margin to a typical benchmark for their industry *in Nigeria* (e.g., 'The average for retail in Nigeria is ~15%...')."
    )
    future_projection: str = Field(
        ...,
        description="A high-level 3-6 month projection, assuming current trends continue (e.g., 'If costs remain stable, you can project...')."
    )
    key_areas_for_improvement: List[str] = Field(
        ...,
        description="A bulleted list of 2-3 specific areas for improvement, based on the data (e.g., 'Variable costs are high...')."
    )
    strategic_growth_advice: str = Field(
        ...,
        description="A concluding paragraph of high-level strategic advice for business growth in the Nigerian context."
    )

    @field_validator('profit_margin_percentage')
    def round_margin(cls, v):
        # Helper to ensure the percentage is neatly rounded.
        return round(v, 2)

# --- 2. System Instruction and LLM Configuration ---
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

SYSTEM_PROMPT = """
You are an expert Nigerian Business Analyst and Consultant. Your goal is to analyze a small business's monthly financial data and provide a high-level, actionable report.

**YOUR TASK:**
1.  You will be given the user's industry, monthly revenue, fixed costs, and variable costs.
2.  Calculate their Net Profit (Revenue - Fixed - Variable) and Profit Margin percentage.
3.  **Crucially:** Use your general knowledge to establish a *reasonable benchmark profit margin* for their specific industry *in Nigeria*.
4.  Compare their performance against this benchmark in the 'industry_benchmark' field.
5.  Analyze their cost structure (fixed vs. variable) relative to their revenue.
6.  Provide a simple 3-6 month future projection in the 'future_projection' field.
7.  Identify the top 2-3 'key_areas_for_improvement' (e.g., "High Variable Costs", "Low Revenue Volume").
8.  Provide 'strategic_growth_advice' focused on growth within the Nigerian market.
9.  You MUST return your entire response as a valid JSON object matching the 'BusinessAnalysisReport' schema. Do NOT include any text outside the JSON.
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
        Monthly Fixed Costs: {user_data.get('fixed_costs'):,.2f} NGN
        Monthly Variable Costs: {user_data.get('variable_costs'):,.2f} NGN
        ---
        Calculated Total Costs: {user_data.get('total_cost'):,.2f} NGN
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
            max_tokens=2048, 
            temperature=0.1 # Low temp for factual analysis
        )

        return report_object

    except Exception as e:
        print(f"\nAn API/Connection error occurred: {e}")
        # Return a fallback object on severe failure
        return BusinessAnalysisReport(
            profit_margin_percentage=0.0,
            performance_summary="Error: Could not generate analysis.",
            industry_benchmark="N/A",
            future_projection="N/A",
            key_areas_for_improvement=["API connection failed. Check AWS credentials and model access."],
            strategic_growth_advice="Unable to provide advice due to a system error. Please ensure your AWS Bedrock environment is configured correctly for Llama 3 70B."
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
    print("This tool collects your monthly financial data to provide analysis and growth advice.")
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
            fixed_costs = get_numeric_input("Enter your total Monthly Fixed Costs (e.g., rent, salaries) (NGN):\n> ")
            variable_costs = get_numeric_input("Enter your total Monthly Variable Costs (e.g., materials, shipping) (NGN):\n> ")
            
            # --- 2. Pre-Calculation ---
            total_cost = fixed_costs + variable_costs
            net_profit = revenue - total_cost
            
            print("\n--- Your Data Summary ---")
            print(f"  Monthly Revenue:    ₦{revenue:,.2f}")
            print(f"  Total Monthly Cost: ₦{total_cost:,.2f}")
            print(f"  Net Profit/Loss:    ₦{net_profit:,.2f}")
            
            if revenue == 0:
                print("\nCannot calculate profit margin or provide analysis with zero revenue.")
                continue

            # --- 3. Get AI Analysis ---
            user_data_payload = {
                "industry": industry,
                "revenue": revenue,
                "fixed_costs": fixed_costs,
                "variable_costs": variable_costs,
                "total_cost": total_cost,
                "net_profit": net_profit
            }
            
            report = get_business_analysis(user_data_payload)
            
            # --- 4. Display Report ---
            print("\n" + "="*50)
            print("| AI BUSINESS ANALYSIS REPORT |")
            print("="*50)

            print("\n--- PERFORMANCE ---")
            print(f"Profit Margin: {report.profit_margin_percentage}%")
            print(f"Summary: {report.performance_summary}")
            print(f"Benchmark: {report.industry_benchmark}")

            print("\n--- PROJECTION & IMPROVEMENT ---")
            print(f"Future Projection: {report.future_projection}")
            print("\nKey Areas for Improvement:")
            for item in report.key_areas_for_improvement:
                print(f"  • {item}")

            print("\n--- STRATEGIC GROWTH ADVICE ---")
            print(report.strategic_growth_advice)
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
