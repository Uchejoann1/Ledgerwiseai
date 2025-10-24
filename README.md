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
