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
# Uses Llama 3 70B Instruct for high performance and stability on on-demand Bedrock throughput.
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
        
        # Llama 3 works best by combining the system prompt and user query into one user message.
        full_query = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {user_query}"

        # 3. Create the structured completion request
        # Llama 3 Instruct models accept a single 'user' message containing the full prompt.
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
        # NOTE: Using exponential backoff would be added here in a production environment
        print(f"An API/Connection error occurred: {e}")
        # Return a fallback object on severe failure
        return BusinessAdvice(
            relevance_score=0.0,
            advice_type="IRRELEVANT",
            advice_title="System Error: API Failure",
            advice="The advisory service is currently unavailable due to a connection or API error. Please check your AWS configuration or model access."
        )

# --- 4. Interactive Execution (Simplified Output in a Loop) ---
if __name__ == "__main__":
    print("\n--- Nigerian Business and Tax Advisor (Interactive) ---")
    #print(f"Model: {MODEL_ID}")
    print("Type 'quit' or 'exit' at any prompt to end the session.\n")
    
    # List of simple greetings to check against
    GREETINGS = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    
    # Variable to hold the query for the next loop iteration
    next_query = None

    while True:
        try:
            # --- PHASE 1: Get Question or Greeting ---
            if next_query:
                # Use the query provided in the continuation prompt
                user_query = next_query
                next_query = None # Clear the next_query for the next turn
                print("-" * 50)
            else:
                # Primary prompt
                user_input = input("Ask a business or tax question (specific to Nigeria):\n> ")
                user_query = user_input.strip()

            # Check for immediate exit command
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
                break
                
            if not user_query:
                continue
            
            # Check if the input is a simple greeting
            if user_query.lower() in GREETINGS:
                print("\nHello! I am your Nigerian Business and Tax Advisor. How can I assist you with your business strategy or tax compliance questions today?")
                print("-" * 50)
                continue # Go back to the start of the loop to ask for a specific question

            
            # --- PHASE 2: Get Advice ---
            advice_object = get_nigerian_advice(user_query)
            
            # Display only the required output
            print("\n" + "="*50)
            print(f"TITLE: {advice_object.advice_title.upper()}")
            print(f"TYPE: {advice_object.advice_type.upper()}")
            print("="*50)
            print(advice_object.advice)
            
            # --- PHASE 3: Check for Continuation ---
            
            continue_prompt = "\nDo you have any other questions about Nigerian business or tax? (Type 'no' to exit, or enter your next question):\n> "
            
            # Read the next input
            next_input = input(continue_prompt).strip()
            
            if not next_input:
                # If they just hit Enter, go back to the main prompt
                continue
            elif next_input.lower() in ['no', 'n', 'quit', 'exit', 'q']:
                # Exit command
                print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
                break
            else:
                # User typed a new question, set it for the next loop iteration
                next_query = next_input
            
        except EOFError:
            # Handle Ctrl+D or Ctrl+Z for graceful exit
            print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
            break
        except Exception as e:
            # Handle unexpected runtime errors gracefully
            print(f"\nAn unexpected runtime error occurred: {e}. Restarting loop.")
            print("\n" + "-"*50 + "\n")
            continue
