> # Ledgerwiseai
> 🇳🇬 Nigerian Business and Tax Advisor (AWS Bedrock & Instructor)
This Python script provides an interactive command-line chatbot specialized in offering structured advice strictly related to Nigerian business operations, regulations, and taxation.
It utilizes the AWS Bedrock service to access a powerful Large Language Model (LLM) and the instructor library to enforce a strict JSON output format, ensuring reliable and structured responses.

Features
-Strictly Focused Advice: Answers only questions related to Nigerian business and tax law.
-Guardrail Enforcement: Automatically detects irrelevant questions and returns a professional rejection message.
-Structured Output: Uses Pydantic and the instructor library to guarantee the output is a clean, machine-readable JSON object.
-Interactive Loop: Allows for continuous conversation until the user decides to exit
-Model: Powered by the high-performance Meta Llama 3 70B Instruct model via AWS Bedrock.

Prerequisites 
Before running this script, you must have the following installed and configured:
-Python: Python 3.8 or higher.
-AWS Credentials: Your AWS credentials must be configured locally (e.g., via the AWS CLI using aws configure) so that boto3 can authenticate.
-Bedrock Access: Ensure the Meta Llama 3 70B Instruct model is enabled for use in your AWS Bedrock console.

Installation
-Install the required Python libraries using pip:pip install boto3 pydantic instructor

Important Note on AWS Region
-The script uses os.environ.get("AWS_REGION", "us-east-1"). If your Bedrock models are hosted in a different region (e.g., us-west-2 or eu-central-1), 
-ensure you either:Set the AWS_REGION environment variable in your terminal, ORChange the default value in the script to your desired region.
> Resources Used
> -Youtube
> -Gemini

🇳🇬 Nigerian Tax & Business Advisor

This is a Python-based command-line tool that uses AI to analyze financial data from a CSV or Excel file. It calculates a company's Nigerian tax liabilities (CIT, TET, and VAT), audits tax payments, and provides both tax compliance recommendations and scannable business growth advice.

This tool is powered by the Llama 3 70B model via AWS Bedrock.

Features

File Input: Reads financial data directly from .csv or .xlsx files.

Comprehensive Tax Calculation:

Profit Tax: Calculates Corporate Income Tax (CIT) based on company size (Medium/Large) and Tertiary Education Tax (TET).

VAT: Calculates Value Added Tax (VAT) remittable to FIRS.

Compliance Audit: Compares the calculated tax due against a "Profit Tax Paid" field from your file to determine if the company is compliant, underpaid, or has overpaid.

Dual-AI Advice:

Tax Compliance: Provides actionable steps for remaining compliant.

Business Growth: Analyzes financial data (e.g., revenue vs. expenses) to give high-level business advice.

Prerequisites

Python 3.8+

An AWS Account: You must have access to AWS Bedrock and have enabled access for the meta.llama3-70b-instruct-v1:0 model.

AWS Credentials: Your AWS access keys must be configured in your environment (e.g., via aws configure or environment variables).

Installation

Clone this repository or save the Python script (nigerian_tax_calculator.py) to your computer.

Install the required Python libraries:

pip install boto3 pydantic instructor pandas openpyxl


Data File Format

Your .csv or .xlsx file must contain columns that can be identified as "metrics" and "amounts." The script is flexible but works best with headers like:

Metric Column: Metric, Item, Description, Particulars

Amount Column: Amount_NGN, Amount, Value, NGN

The script will search for the following keywords in your Metric Column (case-insensitive):

Total Revenue (or Sales) - Mandatory

Cost of Sales

Operating Expenses

Profit Tax Paid (or CIT Paid, Tax Paid)

Output VAT (or VAT on Sales)

Input VAT (or VAT on Purchases)