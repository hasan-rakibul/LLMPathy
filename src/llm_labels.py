import os
from groq import Groq
from typing import Dict, List
import numpy as np
from utils import log_info, log_debug, read_file
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    max_retries=0 # disable retry as we are handling it manually
)

# taken from https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
def as_system(content: str) -> Dict:
    return {"role": "system", "content": content}

def as_assistant(content: str) -> Dict:
    return {"role": "assistant", "content": content}

def as_user(essay: str) -> Dict:
    content = f"Essay: ```{essay}\n```\
        Now, provide scores with respect to Batson's empathy scale. That is, provide scores between 1.0 and 7.0 for each of the following emotions: sympathetic, moved, compassionate, tender, warm and softhearted.\
        You must provide comma-separated floating point scores, where a score of 1.0 means the individual is not feeling the emotion at all, and a score of 7.0 means the individual is extremely feeling the emotion.\
        You must not provide any other outputs apart from the scores."
    return {"role": "user", "content": content}

@retry(wait=wait_exponential(multiplier=1, min=120, max=600), stop=stop_after_attempt(5))
def chat_completion(
    messages: List[Dict],
    model: str = "llama3-70b-8192",
    temperature: float = 0.0, # 0.0 is deterministic
    top_p: float = 0.01,
) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    response_content = response.choices[0].message.content
    
    return response_content
        
def _perse_scores(subscale_scores: str) -> float:
    log_debug(logger, f"Subscale scores: {subscale_scores}")
    score_array = np.array([float(s) for s in subscale_scores.split(",")])
    return np.mean(score_array)

def completion(question: Dict) -> str:
    system = as_system(
        "Your task is to measure the empathy of individuals based on their written essays.\
        You will assess empathy using Batson's definition, which specifically measures how the subject is feeling each of the following six emotions: sympathetic, moved, compassionate, tender, warm and softhearted.\
        Human subjects wrote these essays after reading a newspaper article involving harm to individuals, groups of people, nature, etc. The essay is provided to you within triple backticks."
    )

    prompt = [system]
    
    prompt.append(question)
    log_debug(logger, f"Prompt: {prompt}")

    subscale_scores = chat_completion(
        prompt
    )
    return subscale_scores

def _measure_empathy_LLM(file_path:str) -> None:
    df = read_file(file_path)
    assert df["essay"].isnull().sum() == 0, "There are missing essay in the dataset"
    assert df["essay"].isna().sum() == 0, "There are NA essay in the dataset"
    if "llm_empathy" in df.columns:
        log_info(logger, "llm_empathy exists in columns. So, in resume mode.")
        resume = True
        save_path = file_path # overwrite the file in resume mode
    else:
        resume = False
        if file_path.endswith(".tsv"):
            save_path = file_path.replace(".tsv", "_llama.tsv")
        elif file_path.endswith(".csv"):
            save_path = file_path.replace(".csv", "_llama.tsv")

    for index, row in df.iterrows():
        # skip if already done
        if resume:
            if not np.isnan(row["llm_empathy"]):
                log_info(logger, f"Skipping index {index}, as it has llm_empathy: {row["llm_empathy"]}")
                continue

        question = as_user(
            essay=row["essay"]
        )
        probably_subscale_scores = completion(question)
        try:
            empathy_score = _perse_scores(probably_subscale_scores)
        except:
            log_info(logger, f"Failed to parse scores for index: {index} and essay: {row["essay"][:50]}")
            log_info(logger, f"The failed scores: {probably_subscale_scores}")
            log_info(logger, f"Trying again for the above essay...")

            # ask again as LLMs may give correct answer this time
            question = as_user(
                essay=row["essay"]
            )
            probably_subscale_scores = completion(question)
            try:
                empathy_score = _perse_scores(probably_subscale_scores)
            except:
                log_info(logger, f"Failed to parse scores for index: {index} and essay: {row["essay"][:50]}")
                log_info(logger, f"The failed scores: {probably_subscale_scores}")
                log_info(logger, f"Failed again. So, skipping this essay.")
                empathy_score = np.nan
        
        df.loc[index, "llm_empathy"] = empathy_score
        log_info(logger, f"Index: {index}, Essay: {row['essay'][:50]}..., Empathy score: {empathy_score}")

        # save time to time
        if index % 5 == 0:
            df.to_csv(save_path, sep="\t", index=False)

    # save at the end
    df.to_csv(save_path, sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure empathy using LLM')
    parser.add_argument('--file_path', type=str, help='File path to the dataset')
    args = parser.parse_args()

    _measure_empathy_LLM(args.file_path)
