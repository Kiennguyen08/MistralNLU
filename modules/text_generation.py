import ast
import copy
import html
import random
import time
import traceback
import logging
import torch
from modules.extensions import apply_extensions
import shared
from modules.callbacks import clear_torch_cache


logger = logging.getLogger(__name__)


def apply_stopping_strings(reply, all_stop_strings):
    stop_found = False
    for string in all_stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou:"
        # is completed, trim it
        for string in all_stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


def generate_reply(
    question,
    state,
    stopping_strings=None,
    is_chat=False,
    escape_html=False,
    for_ui=False,
):
    # Find the appropriate generation function
    generate_func = apply_extensions("custom_generate_reply")
    if generate_func is None:
        if shared.model_name == "None" or shared.model is None:
            logger.error("No model is loaded! Select one in the Model tab.")
            yield ""
            return

        if shared.model.__class__.__name__ in [
            "LlamaCppModel",
            "RWKVModel",
            "ExllamaModel",
            "Exllamav2Model",
            "CtransformersModel",
        ]:
            generate_func = generate_reply_custom

    # Prepare the input
    original_question = question
    if not is_chat:
        state = apply_extensions("state", state)
        question = apply_extensions("input", question, state)

    # Find the stopping strings
    all_stop_strings = []
    for st in (stopping_strings, state["custom_stopping_strings"]):
        if type(st) is str:
            st = ast.literal_eval(f"[{st}]")

        if type(st) is list and len(st) > 0:
            all_stop_strings += st

    if shared.args.verbose:
        print(f"\n\n{question}\n--------------------\n")

    shared.stop_everything = False
    clear_torch_cache()
    seed = set_manual_seed(state["seed"])
    last_update = -1
    reply = ""
    is_stream = state["stream"]
    if len(all_stop_strings) > 0 and not state["stream"]:
        state = copy.deepcopy(state)
        state["stream"] = True

    min_update_interval = 0
    if state.get("max_updates_second", 0) > 0:
        min_update_interval = 1 / state["max_updates_second"]

    # Generate
    for reply in generate_func(
        question, original_question, seed, state, stopping_strings, is_chat=is_chat
    ):
        reply, stop_found = apply_stopping_strings(reply, all_stop_strings)
        if escape_html:
            reply = html.escape(reply)
        if is_stream:
            cur_time = time.time()

            # Maximum number of tokens/second
            if state["max_tokens_second"] > 0:
                diff = 1 / state["max_tokens_second"] - (cur_time - last_update)
                if diff > 0:
                    time.sleep(diff)

                last_update = time.time()
                yield reply

            # Limit updates to avoid lag in the Gradio UI
            # API updates are not limited
            else:
                if cur_time - last_update > min_update_interval:
                    last_update = cur_time
                    yield reply

        if stop_found or (state["max_tokens_second"] > 0 and shared.stop_everything):
            break

    if not is_chat:
        reply = apply_extensions("output", reply, state)

    yield reply


def set_manual_seed(seed):
    seed = int(seed)
    if seed == -1:
        seed = random.randint(1, 2**31)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def encode(prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
    if shared.tokenizer is None:
        raise ValueError("No tokenizer is loaded")

    if shared.model.__class__.__name__ in [
        "LlamaCppModel",
        "RWKVModel",
        "CtransformersModel",
        "Exllamav2Model",
    ]:
        input_ids = shared.tokenizer.encode(str(prompt))
        if shared.model.__class__.__name__ not in ["Exllamav2Model"]:
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
    else:
        input_ids = shared.tokenizer.encode(
            str(prompt), return_tensors="pt", add_special_tokens=add_special_tokens
        )
        if not add_bos_token:
            while (
                len(input_ids[0]) > 0
                and input_ids[0][0] == shared.tokenizer.bos_token_id
            ):
                input_ids = input_ids[:, 1:]

    # Handling truncation
    if truncation_length is not None:
        input_ids = input_ids[:, -truncation_length:]

    if (
        shared.model.__class__.__name__
        in [
            "LlamaCppModel",
            "RWKVModel",
            "ExllamaModel",
            "Exllamav2Model",
            "CtransformersModel",
        ]
        or shared.args.cpu
    ):
        return input_ids
    elif shared.args.deepspeed:
        return input_ids.to(device=local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        return input_ids.to(device)
    else:
        return input_ids.cuda()


def generate_reply_custom(
    question, original_question, seed, state, stopping_strings=None, is_chat=False
):
    """
    For models that do not use the transformers library for sampling
    """
    seed = set_manual_seed(state["seed"])

    t0 = time.time()
    reply = ""
    try:
        if not is_chat:
            yield ""

        if not state["stream"]:
            reply = shared.model.generate(question, state)
            yield reply
        else:
            for reply in shared.model.generate_with_streaming(question, state):
                yield reply

    except Exception:
        traceback.print_exc()
    finally:
        t1 = time.time()
        original_tokens = len(encode(original_question)[0])
        new_tokens = len(encode(original_question + reply)[0]) - original_tokens
        print(
            f"Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})"
        )
        return


def get_max_prompt_length(state):
    return state["truncation_length"] - state["max_new_tokens"]


def get_encoded_length(prompt):
    length_after_extensions = apply_extensions("tokenized_length", prompt)
    if length_after_extensions is not None:
        return length_after_extensions

    return len(encode(prompt)[0])
