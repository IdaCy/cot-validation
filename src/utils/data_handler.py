import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Any

def bloomer(n: int = 1) -> None:
    """
    A fun little "flower" print-out function?
    """
    print('(˵◕ ɛ ◕˵✿)' + ' ❀' * n)

def batch_generate(
    df: pd.DataFrame,
    input_column: str = "prompt",
    target_column: str = "label",
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """
    Splits a DataFrame into a list of small batch dictionaries 
    containing 'input', 'target', and 'metadata'.

    Parameters
    ----------
    df : pd.DataFrame - The dataframe containing prompts.
    input_column : str - Column name for the input text.
    target_column : str  Column name for the target/label.
    batch_size : int

    Returns
    -------
    List[Dict[str, Any]] - keys: - ['input', 'target', 'metadata'].
    """
    batches = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i: i + batch_size]
        batch_dict = {
            "input": batch_df[input_column].tolist(),
            "target": batch_df[target_column].tolist(),
            "metadata": {
                col: batch_df[col].tolist() 
                for col in df.columns if col not in [input_column, target_column]
            },
        }
        batches.append(batch_dict)
    return batches


def tokens_generate(
    batches: List[Dict[str, Any]],
    tokenizer,
    device: str = "mps"
) -> List[Dict[str, Any]]:
    """
    Tokenizes the 'input' in each batch using a Hugging Face tokenizer.

    Parameters
    ----------
    batches : List[Dict[str, Any]]
    tokenizer : PreTrainedTokenizer
    device : str

    Returns
    -------
    List[Dict[str, Any]]
        Same length as 'batches', but each item includes 'tokenized_prompts' 
        (the tensors on the specified device).
    """
    tokenized_batches = []
    for batch in batches:
        tokenized_prompts = tokenizer(
            batch["input"],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        tokenized_prompts = {
            k: v.to(device) for k, v in tokenized_prompts.items()
        }

        tokenized_batches.append({
            "input": batch["input"],
            "tokenized_prompts": tokenized_prompts,  # dict of tensors
            "target": batch["target"],
            "metadata": batch["metadata"]
        })

    return tokenized_batches

