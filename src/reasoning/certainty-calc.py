import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

def capture_confidence_tokens(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    generated_tokens: torch.Tensor,
    correct_answer_str: str,
    threshold: float = 0.90,
    topk: int = 1000,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Analyzes the model's token-by-token probabilities to see how soon
    it becomes >= `threshold` confident in the `correct_answer_str`.

    This function replicates the step-by-step 'partial generation + 
    next token distribution' approach you've used in your notebook.

    Parameters
    ----------
    model : torch.nn.Module
    tokenizer : PreTrainedTokenizer
    input_ids : torch.Tensor IDs
    generated_tokens : torch.Tensor - shape [1, num_generated_tokens].
    correct_answer_str : str
    threshold : float - the confidence level
    topk : int - Number of top tokens to compute probabilities for each step 
    device : str - 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    dict
        {
          "step_details": List[dict],  # Probability info at each step
          "step_90_confidence": Optional[int], # The step index at which 
                                              # prob >= threshold (or None)
          "final_answer": str
        }
    """
    # Convert the correct answer string to token ID(s).
    # For now: one token!
    correct_answer_ids = tokenizer.encode(correct_answer_str, add_special_tokens=False)
    if len(correct_answer_ids) != 1:
        raise ValueError("For now, this function handles only single-token answers.")
    correct_token_id = correct_answer_ids[0]

    # inserting a small finishing token so the model doesn't continue generating. 
    replace_str = "</think>\n"
    replace_ids = tokenizer.encode(replace_str, return_tensors="pt").to(device)

    step_details = []
    step_90_confidence = None

    # generated_tokens, shape [1, length]. - removing batch dimension:
    generated_tokens = generated_tokens[0]  # shape = [length]
    
    # Evaluate partial generation at each step i.
    for i, original_token_id in enumerate(generated_tokens):
        # Everything up to (but not including) index i, plus the 'replace_str' to end 
        # the sequence so we can see the next-token distribution.
        partial_input_ids = torch.cat([
            input_ids,
            generated_tokens[:i].unsqueeze(0),
            replace_ids
        ], dim=1)

        with torch.no_grad():
            outputs = model(partial_input_ids)

        # The logits for the *next* token after that partial sequence:
        next_token_logits = outputs.logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)

        # Probability of the correct answer token
        prob_correct_token = probs[0, correct_token_id].item()

        # get top-k tokens for debugging or analysis
        topk_probs, topk_ids = torch.topk(probs, k=topk, dim=-1)
        topk_info = []
        for rank in range(topk):
            token_id = topk_ids[0, rank].item()
            token_prob = topk_probs[0, rank].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            topk_info.append({
                "token": token_str,
                "probability": float(token_prob),
                "rank": rank
            })

        step_details.append({
            "step": i,
            "prev_token_str": tokenizer.decode([original_token_id], skip_special_tokens=True),
            "prob_correct_token": prob_correct_token,
            "topk_tokens": topk_info
        })

        # If we exceed threshold, record the step and stop checking:
        if step_90_confidence is None and prob_correct_token >= threshold:
            step_90_confidence = i
            # We can continue collecting steps or break immediately!
            # If want the *first* time it crosses threshold:
            # break

    final_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {
        "step_details": step_details,
        "step_90_confidence": step_90_confidence,
        "final_answer": final_answer
    }

