from pathlib import Path
from typing import Literal
from IPython.display import display
import pandas as pd
import torch
import einops

from jaxtyping import Float
from tqdm import tqdm
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)
from transformer_lens import HookedTransformer
from collections import defaultdict
import numpy as np
from scipy.stats import mode
import circuitsvis as cv

from data_utils import load_prompts_from_msmarco_samples_from_rag_truth


def identify_copying_heads(model: HookedTransformer, min_score: float = 0.6) -> dict[int, list[int]]:
    batch_size = 10
    seq_len = 50
    size = (batch_size, seq_len)
    input_tensor = torch.randint(1000, 10000, size)

    random_tokens = input_tensor.to(model.cfg.device)
    repeated_tokens = einops.repeat(random_tokens, "batch seq_len -> batch (2 seq_len)")

    induction_score_store = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    def induction_score_hook(
        pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
    ):
        # We take the diagonal of attention paid from each destination position to source positions seq_len-1 tokens back
        # (This only has entries for tokens with index>=seq_len)
        induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1-seq_len)
        # Get an average score per head
        induction_score = einops.reduce(induction_stripe, "batch head_index position -> head_index", "mean")
        # Store the result.
        induction_score_store[hook.layer(), :] = induction_score

    # We make a boolean filter on activation names, that's true only on attention pattern names.
    pattern_hook_names_filter = lambda name: name.endswith("pattern")

    model.run_with_hooks(
        repeated_tokens, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(
            pattern_hook_names_filter,
            induction_score_hook
        )]
    )

    layer_inds, head_inds = np.where(induction_score_store.cpu().numpy() > min_score)

    induction_layer_to_head_map = defaultdict(list)
    for layer, head in zip(layer_inds, head_inds):
        induction_layer_to_head_map[layer].append(head)

    return induction_layer_to_head_map


def get_model_attention_patterns(model, input: torch.Tensor, induction_layer_to_head_map: dict[int, list[int]], visualise: bool = False) -> dict[str, np.ndarray]:
    model.reset_hooks()

    pattern_store = {}
    def save_attention_pattern_hook(
        pattern: Float[torch.Tensor, "batch head_index dest_pos source_pos"],
        hook: HookPoint,
    ):
        layer_ind = hook.layer()
        for head_ind in induction_layer_to_head_map[layer_ind]:
            p = pattern[0, head_ind, :, :].detach().cpu().numpy()
            pattern_store[f"layer_{layer_ind}_head_{head_ind}"] = p

            if visualise:
                display(
                    cv.attention.attention_patterns( # type: ignore
                        tokens=model.to_str_tokens(input, ), 
                        attention=pattern[0, head_ind, :, :].detach().cpu().numpy()[None, :, :] # Add a dummy axis, as CircuitsVis expects 3D patterns.
                    )
                )
    model.run_with_hooks(
        input, 
        return_type=None, 
        fwd_hooks=[
            (utils.get_act_name("pattern", layer), save_attention_pattern_hook) for layer in induction_layer_to_head_map.keys()
        ]
    )

    model.reset_hooks()

    return pattern_store


AttnPatternFusionOptions = Literal["majority_vote", "majority_vote_without_zeros", "best_scoring_head", "most_attending_head"]

def get_attended_tokens(pattern_store: dict, fusion_approach: AttnPatternFusionOptions = "most_attending_head") -> np.ndarray:
    seq_len = next(iter(pattern_store.values())).shape[0]

    max_token_attention = np.zeros((len(pattern_store), seq_len), dtype=int)

    for i, pattern in enumerate(pattern_store.values()):
        max_token_attention[i] = pattern.argmax(axis=1)
    
    if fusion_approach == "majority_vote":
        fused_res = mode(max_token_attention, axis=0).mode
    elif fusion_approach == "majority_vote_without_zeros":
        # we can't set np.nan to interger array
        token_attentions = max_token_attention.astype(np.float32)
        token_attentions[token_attentions == 0] = np.nan
        fused_res = mode(token_attentions, axis=0, nan_policy="omit").mode
        fused_res[np.isnan(fused_res)] = 0
    elif fusion_approach == "best_scoring_head":
        fused_res = max_token_attention[5, :]
    elif fusion_approach == "most_attending_head":
        most_attended_head_ind = np.argmax((max_token_attention != 0).sum(axis=1))
        fused_res = max_token_attention[most_attended_head_ind, :]


    return fused_res


def build_citation_str(
        model, 
        prompt_tokens: list[int], 
        generated_tokens: list[int], 
        generated_token_attention_res: list[int],
        # This can backfire in case of specific facts, i.e. which occupy a single token.
        minimal_token_count_in_citation: int = 3
    ) -> str:
    generated_text_with_citations = []
    last_attented_token_ind = -100
    current_citation = []
    complete_citations = []

    def save_citation_and_restart():
        nonlocal current_citation, complete_citations, generated_text_with_citations
        if len(current_citation) >= minimal_token_count_in_citation:
            complete_citations.append(current_citation)
            citation_ind = len(complete_citations)
            citation_refer_tokens = model.to_tokens(f"[{citation_ind}]", prepend_bos=False).cpu().squeeze().numpy().tolist()
            generated_text_with_citations.extend(citation_refer_tokens)
        current_citation = []

    for token, attended_token_ind in zip(generated_tokens, generated_token_attention_res):
        generated_text_with_citations.append(token)

        allowed_citation = attended_token_ind != 0 and attended_token_ind < len(prompt_tokens)
        continue_citation = allowed_citation and attended_token_ind - last_attented_token_ind in list(range(1, 5))
        citation_in_progress = len(current_citation) > 0

        if allowed_citation:
            if citation_in_progress and continue_citation:
                current_citation.append(prompt_tokens[attended_token_ind])
            elif citation_in_progress and not continue_citation:
                save_citation_and_restart()
            else:
                current_citation = [prompt_tokens[attended_token_ind]]
        else:
            if citation_in_progress:
                save_citation_and_restart()

        last_attented_token_ind = attended_token_ind

    generated_text_with_citations_str = model.to_string(generated_text_with_citations)
    citation_texts = "\n\n"
    for i, citation_tokens in enumerate(complete_citations):
        citation_str = model.to_string(citation_tokens)
        citation_str = citation_str.strip()
        citation_texts += f"[{i+1}]: {citation_str}\n"
    generated_text_with_citations_str += citation_texts

    return generated_text_with_citations_str


def _median_filter_1d_array(arr: list[int]):
    # [..., 123, 124, 0, 126, 127, ...] -> [..., 123, 124, 125, 126, 127, ...]
    for i in range(1, len(arr) - 1):
        if arr[i] == 0 and arr[i+1] - arr[i-1] == 2:
                arr[i] = arr[i-1] + 1


def generate_with_citations(model, prompt: str, induction_layer_to_head_map: dict[int, list[int]], seed: int = 7575) -> str:
    torch.manual_seed(seed)
    prompt_plus_generated = model.generate(prompt, max_new_tokens=256, temperature=1, do_sample=True)

    prompt_plus_generated_tensor = model.to_tokens(prompt_plus_generated).to(model.cfg.device)
    pattern_store = get_model_attention_patterns(model, prompt_plus_generated_tensor, induction_layer_to_head_map)
    attended_tokens = get_attended_tokens(pattern_store)

    prompt_tokens = model.to_tokens(prompt)[0].cpu().numpy()
    prompt_token_count = len(prompt_tokens)
    generated_tokens = prompt_plus_generated_tensor.squeeze().cpu().numpy()[prompt_token_count:]
    generated_tokens_attendance = attended_tokens[prompt_token_count:]
    assert len(generated_tokens) == len(generated_tokens_attendance)

    # for easier processing
    generated_tokens, prompt_tokens,  generated_tokens_attendance = ([int(val) for val in arr] for arr in (generated_tokens, prompt_tokens, generated_tokens_attendance))
    _median_filter_1d_array(generated_tokens_attendance)
    generated_text_with_citations = build_citation_str(model, prompt_tokens, generated_tokens, generated_tokens_attendance)

    return generated_text_with_citations


def main():
    torch.set_grad_enabled(False)
    device = utils.get_device()
    model = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device=device, dtype="float16")

    induction_layer_to_head_map = identify_copying_heads(model)
    
    dataset_path = Path(__file__).parent / "RAGTruth-main/dataset/"
    prompts = load_prompts_from_msmarco_samples_from_rag_truth(dataset_path)

    prompts_inds = [1, 244]

    results = []
    for prompt_ind in tqdm(prompts_inds):
        prompt = prompts[prompt_ind]
        generation_with_citations = generate_with_citations(model, prompt, induction_layer_to_head_map)
        print(f"prompt_ind: {prompt_ind}\nPROMPT:\n{prompt}\nANSWER:\n{generation_with_citations}\n\n")
        results.append({
            "prompt_ind": prompt_ind,
            "prompt": prompt,
            "generation": generation_with_citations
        })
    pd.DataFrame(results).to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()