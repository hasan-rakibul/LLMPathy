import os
import transformers
import torch

def llm_groq():
    from groq import Groq
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "What is empathy?"
            }
        ],
        model="llama3-70b-8192"
    )

    print(chat_completion.choices[0].message.content)


def authenticate_huggingface():
    from huggingface_hub import login
    login(token=os.environ.get("HF_API_KEY"))

def llm_llama():
    # authenticate_huggingface()

    model_id = "meta-llama/Meta-Llama-3.1-70B"

    pipeline = transformers.pipeline(
        "text-generation", model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        max_new_tokens=20
    )
    print(pipeline("Hey how are you doing today?"))

"""
prompt = [
    {
        "role": "system",
        "content": f"You are an AI model that annotates written essays to provide an empathy score between 1.0 to 7.0 based on the definition of empathy.
         The essays were written by human participants after reading a newspaper article involving harm to individuals, groups of people, nature, etc. 
         The essay is provided to you within triple backticks. 
         Your response must contain one and only empathy score."
    }
]
seed_index = [0, 7, 23]

for index in seed_index:
    prompt.append({
        "role": "user",
        "content": f"Essay: ```{train.loc[index, 'demographic_essay']}```"
    })
    prompt.append({
        "role": "assistant",
        "content": f"{train.loc[index, 'empathy']:.1f}"
    })
"""

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7" # seems only the first one is being used; need to investigate
    # llm_groq()
    llm_llama()