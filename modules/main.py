import traceback
from modules.chat import generate_chat_prompt, get_stopping_strings
from modules.models import unload_model
from modules.models_settings import get_model_metadata
from modules.models import load_model
from modules.text_generation import generate_reply
import shared
import logging


logger = logging.getLogger()


def load_model_wrapper(selected_model, loader, autoload=False):
    if not autoload:
        yield f'The settings for `{selected_model}` have been updated.\n\nClick on "Load" to load it.'
        return

    if selected_model == "None":
        yield "No model selected"
    else:
        try:
            yield f"Loading `{selected_model}`..."
            unload_model()
            if selected_model != "":
                shared.model, shared.tokenizer = load_model(selected_model, loader)

            if shared.model is not None:
                output = f"Successfully loaded `{selected_model}`."

                settings = get_model_metadata(selected_model)
                if "instruction_template" in settings:
                    output += '\n\nIt seems to be an instruction-following model with template "{}". In the chat tab, instruct or chat-instruct modes should be used.'.format(
                        settings["instruction_template"]
                    )

                yield output
            else:
                yield f"Failed to load `{selected_model}`."
        except:
            exc = traceback.format_exc()
            logger.error("Failed to load the model.")
            print(exc)
            yield exc.replace("\n", "\n\n")


if __name__ == "__main__":
    MODEL = "/home/kiennt/vin-project/anybot/models/mistral-7b-v0.1.Q5_K_S.gguf"
    load_model_wrapper(selected_model=MODEL, loader="llama.cpp", autoload=True)
    state = {"history": {"internal": []}}
    prompt = generate_chat_prompt(user_input="how to make a cake", state=state)
    stopping_strings = get_stopping_strings(state)
    for j, reply in enumerate(
        generate_reply(
            prompt,
            state=state,
            stopping_strings=stopping_strings,
            is_chat=True,
            for_ui=False,
        )
    ):
        print(reply)
