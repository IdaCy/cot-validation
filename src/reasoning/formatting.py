import os
import yaml
from typing import List, Union, Dict, Any

def apply_instruct_format(
    inputs: List[str],
    config: Dict[str, Any],
    model: str,
    is_thinking: bool = True,
    is_answer_format: bool = False
) -> List[str]:
    """
    Modify prompts to incorporate DeepSeek usage recommendations:
      - Optionally insert a line that encourages 'step-by-step' reasoning 
        (the <think> tag).
      - Optionally include an instruction that final answers must appear 
        within \\boxed{}.

    Parameters
    ----------
    inputs : List[str]
    config : dict
    model : str
    is_thinking : bool
    is_answer_format : bool

    Returns
    -------
    List[str] - Formatted prompts, each a single string that you can pass to the model.
    """
    # Decide on instructions for final answer
    if is_answer_format:
        answer_format = (
            "Answer the below question. It's EXTREMELY IMPORTANT that you only output a final "
            "response after you've finished thinking, WITHOUT ANY ADDITIONAL WORDS OR PHRASES. "
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
        )
    else:
        answer_format = ""

    # Tag that helps prompt the model to do internal reasoning
    if is_thinking:
        think_tag = "<think>\n"
    else:
        think_tag = ""

    # Fetch the appropriate template from config
    model_config = config["models"][model]
    template = model_config["format"]["template"]

    formatted_inputs = []
    for input_text in inputs:
        out = template.format(
            content=input_text,
            answer_format=answer_format,
            think_tag=think_tag
        )
        formatted_inputs.append(out)

    return formatted_inputs

