from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
)
import torch
import argparse
from mxq.quantize.quant_func import QuantConfig
from mxq.quantize.quantizer import make_quant_linear

import datetime
import tqdm
from torch import nn


def print_time(print_str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {print_str}")


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)

# quantization config
parser.add_argument("--w_bit", type=int, default=16)
parser.add_argument(
    "--w_mode", type=str, choices=["mxfp", "mxem", "mxes"], default=None
)
parser.add_argument("--a_bit", type=int, default=16)
parser.add_argument(
    "--a_mode", type=str, choices=["mxfp", "mxem", "mxes"], default=None
)
parser.add_argument("--group_size", type=int, default=-1)

args = parser.parse_args()


# build model and tokenizer
def build_model_and_enc(model_path):
    print(f"* Building model {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    # fp16 to quantized
    kwargs = {"device_map": "balanced", "torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    print_time("Start pseudo quantize")

    quant_config = QuantConfig(
        w_bit=args.w_bit,
        w_mode=args.w_mode,
        a_bit=args.a_bit,
        a_mode=args.a_mode,
        group_size=args.group_size,
    )
    make_quant_linear(model, quant_config)
    print_time("Finish pseudo quantize")

    return model


def main():

    print("\nargs:", args, "\n")

    model = build_model_and_enc(args.model_path)

    if args.tasks is not None:
        if args.tasks in ["wikitext", "c4", "ptb"]:
            # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
            from .utils.dataload_utils import get_loaders

            model.seqlen = 2048
            _, testenc = get_loaders(
                args.tasks, model=args.model_path, seqlen=model.seqlen
            )

            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

        else:
            # do other evaluations
            lm_eval_model = HFLM(pretrained=model, batch_size=args.batch_size)
            print_time("Start a task")
            task_names = args.tasks.split(",")

            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                num_fewshot=args.num_fewshot,
            )
            print_time("Task finish!")
            print(make_table(results))


if __name__ == "__main__":
    main()
