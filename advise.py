import os
import json
import boto3
from pydantic import BaseModel, Field
from typing import Literal, List
import instructor
from instructor import Mode

# --- 1. Define the (NEW) Structured Output Schema ---
# This new, more detailed model forces the AI to be more comprehensive.
class DetailedBusinessAdvice(BaseModel):
    """
    Structured advice response from the Nigerian Business Advisor. 
    This model defines the exact detailed format the LLM must return.
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
    
    # --- NEW DETAILED FIELDS ---
    key_points_summary: str = Field(
        ...,
        description="A 1-2 sentence concise summary of the most critical part of the advice. If irrelevant, this should state the rejection reason."
    )
    detailed_explanation: str = Field(
        ...,
        description="A comprehensive, multi-paragraph explanation that answers the user's query in depth. Must be well-structured and easy to understand."
    )
    actionable_steps: List[str] = Field(
        ...,
        description="A bulleted list of 2-4 specific, scannable next steps the user should take based on the advice. If irrelevant, this should be an empty list []."
    )
    potential_risks_or_considerations: str = Field(
        ...,
        description="A 1-2 sentence warning about potential risks, pitfalls, or important considerations the user must keep in mind. If irrelevant, this should be 'N/A'."
    )

# --- 2. System Instruction and LLM Configuration ---
MODEL_ID = "meta.llama3-70b-instruct-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1") 

# --- UPDATED SYSTEM PROMPT ---
SYSTEM_PROMPT = f"""
You are a *Senior Expert* Nigerian Business and Tax Consultant. Your goal is to provide *deeply detailed, structured, and comprehensive* advice to help businesses in Nigeria.

You MUST format your entire response using the provided JSON schema. You must fill all fields.

**STRICT RULE & GUARDRAIL:**
If a question is NOT explicitly and solely related to Nigerian business or tax matters, you MUST enforce the guardrail:
- Set 'relevance_score' to 0.0
- Set 'advice_type' to 'IRRELEVANT'
- Set 'advice_title' to 'Query Irrelevant'
- Set 'key_points_summary' to the rejection message: 'I am only programmed to provide business and tax advice specific to Nigeria. Please ask a relevant question.'
- Set 'detailed_explanation' to 'N/A'
- Set 'actionable_steps' to an empty list []
- Set 'potential_risks_or_considerations' to 'N/A'

For relevant queries, ensure 'relevance_score' is 1.0 and all fields are filled with detailed, expert advice.
"""

# --- 3. Main Function to Interact with Bedrock ---
def get_nigerian_advice(user_query: str) -> DetailedBusinessAdvice:
    """
    Calls the Llama 3 70B model via AWS Bedrock using instructor for structured output.
    """
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=AWS_REGION
        )

        client = instructor.from_bedrock(
            bedrock_client,
            mode=Mode.BEDROCK_JSON 
        )
        
        full_query = f"{SYSTEM_PROMPT}\n\nUSER QUERY: {user_query}"

        advice_object = client.messages.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": full_query} # Combined prompt
            ],
            response_model=DetailedBusinessAdvice, # <-- Using the new detailed model
            max_tokens=2048, 
            temperature=0.1 # Keep temperature low for factual advice
        )

        return advice_object

    except Exception as e:
        print(f"An API/Connection error occurred: {e}")
        # Return a fallback object on severe failure
        return DetailedBusinessAdvice(
            relevance_score=0.0,
            advice_type="IRRELEVANT",
            advice_title="System Error: API Failure",
            key_points_summary="The advisory service is currently unavailable due to a connection or API error.",
            detailed_explanation="Please check your AWS configuration or model access.",
            actionable_steps=[],
            potential_risks_or_considerations="N/A"
        )

# --- 4. Interactive Execution (Simplified Output in a Loop) ---
if __name__ == "__main__":
    print("\n--- Nigerian Business and Tax Advisor (Detailed) ---")
    print("Type 'quit' or 'exit' at any prompt to end the session.\n")
    
    GREETINGS = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
    next_query = None

    while True:
        try:
            # --- PHASE 1: Get Question or Greeting ---
            if next_query:
                user_query = next_query
                next_query = None # Clear the next_query
                print("-" * 50)
            else:
                user_input = input("Ask a business or tax question (specific to Nigeria):\n> ")
                user_query = user_input.strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
                break
            
            if not user_query:
                continue
            
            if user_query.lower() in GREETINGS:
                print("\nHello! I am your Nigerian Business and Tax Advisor. How can I assist you with your business strategy or tax compliance questions today?")
                print("-" * 50)
                continue 

            
            # --- PHASE 2: Get Advice ---
            advice_object = get_nigerian_advice(user_query)
            
            # --- UPDATED DISPLAY FOR DETAILED ADVICE ---
            print("\n" + "="*50)
            print(f"TITLE: {advice_object.advice_title.upper()}")
            print(f"TYPE: {advice_object.advice_type.upper()}")
            print("="*50)
            
            if advice_object.relevance_score > 0.5:
                # Print the full, structured report
                print("\n--- KEY SUMMARY ---")
                print(advice_object.key_points_summary)
                
                print("\n--- DETAILED EXPLANATION ---")
                print(advice_object.detailed_explanation)
                
                if advice_object.actionable_steps:
                    print("\n--- ACTIONABLE NEXT STEPS ---")
                    for step in advice_object.actionable_steps:
                        print(f"  • {step}")
                
                print("\n--- KEY CONSIDERATIONS ---")
                print(advice_object.potential_risks_or_considerations)
            else:
                # Print only the rejection message
                print(advice_object.key_points_summary)

            
            # --- PHASE 3: Check for Continuation ---
            continue_prompt = "\n\nDo you have any other questions? (Type 'no' to exit, or enter your next question):\n> "
            next_input = input(continue_prompt).strip()
            
            if not next_input:
                continue
            elif next_input.lower() in ['no', 'n', 'quit', 'exit', 'q']:
                print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
                break
            else:
                next_query = next_input
            
        except EOFError:
            print("\nThank you for using the Nigerian Business Advisor. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected runtime error occurred: {e}. Restarting loop.")
            print("\n" + "-"*50 + "\n")
            continue