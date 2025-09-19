from vllm import LLM, SamplingParams
import logging

logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


model = "EleutherAI/pythia-70m"

llm = LLM(
    model=model,
    dtype="float16",            # use fp16, not bf16
    gpu_memory_utilization=0.70,# target ~70% of VRAM (fits < 2.8 GiB on a 4 GiB card)
    max_model_len=512,          # shrink KV cache
    enforce_eager=True          # avoid large CUDA graph captures
)

sp = SamplingParams(max_tokens=8, logprobs=5)
outs = llm.generate(["Hello from vLLM"], sp)
print(outs[0].outputs[0].logprobs)
