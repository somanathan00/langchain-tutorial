from langchain_community.llms import OpenAI 
import os
from constants import openai_key

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = openai_key

# Initialize OpenAI language model
llm = OpenAI(temperature=0.4)

# Example text prompt
text = "what is the capital of India"

# Generate response
response = llm.predict(text)

print(response)
