import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY not found.")
    exit()

print("Groq API key found.")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {
            "role": "user",
            "content": "Say hello and confirm that the Groq API is working."
        }
    ],
    temperature=0
)

print("\nGroq response:")
print(response.choices[0].message.content)