from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv(override=True)

# Set OPENAI_API_KEY in your environment
client = OpenAI(api_key=os.getenv("TEST_KEY"))

resp = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": "Answer in one short sentence: what is gravity?"}],
    logprobs=True,         # return per-token logprobs
    top_logprobs=5,        # include top-5 alternatives for each token
    max_tokens=50,
)

choice = resp.choices[0]
print("Assistant:", choice.message.content)

print("\nToken logprobs:")
for tok in choice.logprobs.content:
    print(f" token={tok.token!r:>12}  logprob={tok.logprob:.4f}")
    if tok.top_logprobs:
        alts = ", ".join(f"{t.token!r}:{t.logprob:.4f}" for t in tok.top_logprobs)
        print(f"   top: {alts}")
