import os
import json
import boto3
from pydantic import BaseModel, Field
from typing import Literal
import instructor
from instructor import Mode

# --- 1. Define the Structured Output Schema (Pydantic Model) ---
class BusinessAdvice(BaseModel):
    """
    Structured advice response from the Nigerian Business Advisor. 
    This model defines the exact format the LLM must return.
    """
    relevance_score: float = Field(
        ..., 
        description="A score from 0.0 to 1.0 indicating how relevant the query is to Nigerian business or tax law. 1.0 for perfect relevance, 0.0 for irrelevant topics."
    )
    advice_type: Literal["BUSINESS_STRATEGY", "TAX_COMPLIANCE", "IRRELEVANT"] = Field(
        ...,
        description="Categorizes the advice. Use 'IRRELEVANT' if the score is 0.0."
    )
    advice_title: str = Field(
        ..., 
        description="A concise, professional title summarizing the advice provided. If the relevance_score is 0.0, this must be 'Query Irrelevant'."
    )
    advice: str = Field(
        ..., 
        description="The detailed, actionable business or tax advice specific to the Nigerian context. If the relevance_score is 0.0, this should contain the rejection message: 'I am only programmed to provide business and tax advice specific to Nigeria. Please ask a relevant question.' Ensure the output is professional."
    )

# --- 2. System Instruction and LLM Configuration ---
# FIX: Switched to Meta Llama 3 70B Instruct to avoid Anthropic's external use case form requirement.
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")  # Use your appropriate region

# NOTE: The system prompt is combined with the user query for Llama 3 models on Bedrock.
SYSTEM_PROMPT = f"""
You are a professional business and tax advisor specializing *only* in Nigerian business operations, regulations, and taxation (FIRS, states, local governments). Your goal is to provide specific, actionable advice to help businesses in Nigeria improve, grow, and maintain compliance.

You MUST format your entire response using the provided JSON schema.

**STRICT RULE & GUARDRAIL:**
If a question is NOT explicitly and solely related to Nigerian business or tax matters, you MUST enforce the guardrail by setting the 'relevance_score' to 0.0, 'advice_type' to 'IRRELEVANT', 'advice_title' to 'Query Irrelevant', and setting the 'advice' field to the following exact rejection message: 'I am only programmed to provide business and tax advice specific to Nigeria. Please ask a relevant question.'
For relevant queries, ensure 'relevance_score' is 1.0.
"""

# --- 3. Main Function to Interact with Bedrock ---
def get_nigerian_advice(user_query: str) -> BusinessAdvice:
    """
    Calls the Llama 3 70B model via AWS Bedrock using instructor for structured output.
    """
    try:
        # 1. Initialize Bedrock Boto3 Client
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=AWS_REGION
        )

        # 2. Patch the client with instructor
        # Llama 3 is reliable with BEDROCK_JSON mode for structured output.
        client = instructor.from_bedrock(
            bedrock_client,
            mode=Mode.BEDROCK_JSON 
        )
        
        # Lama 3 works best by combining the system prompt and user query into one user message.
        full_query = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {user_query}"

        # 3. Create the structured completion request
        advice_object = client.messages.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": full_query} # Combined prompt
            ],
            response_model=BusinessAdvice,
            max_tokens=2048, 
            temperature=0.0
        )

        return advice_object

    except Exception as e:
        # Keep this print for essential debugging if a connection or API error occurs
        print(f"An API/Connection error occurred: {e}")
        # Return a fallback object on severe failure
        return BusinessAdvice(
            relevance_score=0.0,
            advice_type="IRRELEVANT",
            advice_title="System Error: API Failure",
            advice="The advisory service is currently unavailable due to a connection or API error. Please check your AWS configuration or model access."
        )

# --- 4. Interactive Execution (Simplified Output) ---
if __name__ == "__main__":
    print("\n--- Nigerian Business and Tax Advisor ---")
    print("Model: Meta Llama 3 70B Instruct")
    
    # Get user input
    user_query = input("Ask a business or tax question (specific to Nigeria):\n> ")
    
    if not user_query.strip():
        print("Query cannot be empty. Exiting.")
    else:
        # Get advice
        advice_object = get_nigerian_advice(user_query)
        
        # Display only the required output
        print("\n" + "="*50)
        print(f"{advice_object.advice_title.upper()}")
        print("="*50)
        print(advice_object.advice)
