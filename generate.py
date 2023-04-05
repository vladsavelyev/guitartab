import datasets, transformers
from transformers import (
    GPT2LMHeadModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
    trainer_utils,
)
from model import load_model, load_tokenizer, load_generation_config
from gp_to_tex import alphatex_to_song
from pathlib import Path
import guitarpro as gp


SONG_TMPL = """\
\\title "{}"
.
\\track
\\instrument 34
\\tuning G3 D3 A2 E2
{}
"""


def generate_song(
    out_dir: Path,
    title: str,
    model=None,
    tokenizer=None,
    device=None,
    **kwargs,
):
    if model is None:
        model = load_model()
    if tokenizer is None:
        tokenizer = load_tokenizer()
    if device is None:
        device = "cuda" if transformers.utils.is_torch_cuda_available() else "cpu"

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_config=load_generation_config(),
        **kwargs,
    )

    for i, result in enumerate(generator([tokenizer.bos_token])[0]):
        tex = result["generated_text"]
        tex = tex.replace(tokenizer.eos_token, "")
        tex = tex.rsplit("|", 1)[0]
        tex = SONG_TMPL.format(f'{title} {i}', tex)
        print("-" * 80)
        print(tex)
        print("-" * 80)
        try:
            song = alphatex_to_song(tex)
        except Exception as e:
            print("Could not parse the tex to GP:", e)
        else:
            try:
                fname = "".join(c if c.isalnum() else "_" for c in title).rstrip("_")
                path = out_dir / f"{fname}_{i}.gp"
                gp.write(song, str(path))
            except Exception as e:
                print("Could not write the GP file:", e)
            else:
                print("Saved song to", path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(".").resolve())
    parser.add_argument("--title", type=str, default="Untitled")
    args = parser.parse_args()
    generate_song(args.out_dir, args.title)
