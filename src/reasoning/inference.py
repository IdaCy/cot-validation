import torch
import time
from tqdm import tqdm
from typing import Any, Dict, List

def run_inference(
    model,
    tokenized_batches: List[Dict[str, Any]],
    tokenizer,
    time_tracking: bool = True,
    max_new_tokens: int = 4096
) -> List[Dict[str, Any]]:
    """
    Perform inference on tokenized data.

    Parameters
    ----------
    model : torch.nn.Module
    tokenized_batches : List[Dict[str, Any]]
    tokenizer : PreTrainedTokenizer
    time_tracking : bool
    max_new_tokens : int

    Returns
    -------
    List[Dict[str, Any]] - model responses, decoding, targets, and optional timing info.
    """
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(tokenized_batches), 
                             desc='inference', total=len(tokenized_batches)):
            if time_tracking:
                start_time = time.time()

            response = model.generate(
                **batch['tokenized_prompts'],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False  # or True, depends
            )

            if time_tracking:
                inference_time = time.time() - start_time
            else:
                inference_time = None

            decoded_response = tokenizer.batch_decode(
                response, skip_special_tokens=True
            )

            outputs.append({
                "batch_idx": i,
                "input": batch["input"],
                "tokenized_prompts": batch["tokenized_prompts"],
                "response": decoded_response,
                "target": batch.get("target", None),
                "metadata": batch.get("metadata", {}),
                "inference_time": inference_time
            })

    return outputs
