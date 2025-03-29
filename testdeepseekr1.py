# Assume openai>=1.0.0
from openai import OpenAI
import time

# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key="uNyNN8nhZy9315qCubAHFppqmQmdRSQt",
    base_url="https://api.deepinfra.com/v1/openai",
)

# Start the timer
start_time = time.time()

chat_completion = openai.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": """Brian goes fishing twice as often as Chris, but catches 2/5 times fewer fish than Chris per trip. If Brian caught 400 fish every time he went fishing, how many fish did they catch altogether if Chris went fishing 10 times?"""}],
    max_tokens=1000000,
)

# Calculate execution time
execution_time = time.time() - start_time

print(chat_completion.choices[0].message.content)
print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
print(f"Execution time: {execution_time:.2f} seconds")