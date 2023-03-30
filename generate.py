import argparse
import copy
import os
import sys
import warnings
from typing import Optional, List, Callable

import gradio as gr
import torch
import torch.distributed as dist
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import (
    AlignDevicesHook,
    add_hook_to_module,
    remove_hook_from_submodules,
)
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from peft import PeftModelForCausalLM, LoraConfig
from peft.utils import PeftType, set_peft_model_state_dict
from torch import nn
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.generation.beam_search import BeamSearchScorer
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerationMixin,
)

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


class SteamGenerationMixin(PeftModelForCausalLM, GenerationMixin):
    # support for streamly beam search
    @torch.no_grad()
    def stream_generate(
            self,
            input_ids: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[
                Callable[[int, torch.Tensor], List[int]]
            ] = None,
            **kwargs,
    ):
        self._reorder_cache = self.base_model._reorder_cache
        if is_deepspeed_zero3_enabled() and dist.world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False

        if kwargs.get("attention_mask", None) is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(
                kwargs["input_ids"].shape[0], self.peft_config.num_virtual_tokens
            ).to(kwargs["input_ids"].device)
            kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, kwargs["attention_mask"]), dim=1
            )
        if kwargs.get("position_ids", None) is not None:
            warnings.warn(
                "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
            )
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn(
                "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
            )
            kwargs["token_type_ids"] = None

        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)

        bos_token_id, eos_token_id, pad_token_id = (
            generation_config.bos_token_id,
            generation_config.eos_token_id,
            generation_config.pad_token_id,
        )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = (
                kwargs.get("max_length") is None
                and generation_config.max_length is not None
        )
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = (
                    generation_config.max_new_tokens + input_ids_seq_length
            )
        if generation_config.min_new_tokens is not None:
            generation_config.min_length = (
                    generation_config.min_new_tokens + input_ids_seq_length
            )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = (
                "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            )

        # 2. Set generation parameters if not already defined
        logits_processor = (
            logits_processor if logits_processor is not None else LogitsProcessorList()
        )
        stopping_criteria = (
            stopping_criteria
            if stopping_criteria is not None
            else StoppingCriteriaList()
        )
        # 8. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )
        # 9. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        # 10. go into beam search generation modes
        # 11. prepare beam search scorer
        num_beams = generation_config.num_beams
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=input_ids.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # beam_search logits
        batch_beam_size, cur_len = input_ids.shape
        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )
        beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=input_ids.device
        )
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(
                    0.0 if this_peer_finished else 1.0
                ).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) hack: adjust tokens for Marian.
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[
                                                              :, None
                                                              ].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(
                batch_size, num_beams * vocab_size
            )

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size
            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=None,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat(
                [input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            # increase cur_len
            cur_len = cur_len + 1

            yield input_ids

            if beam_scorer.is_done or stopping_criteria(input_ids, None):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        final_result = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=None,
        )
        yield final_result["sequences"]

    # default it call `model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config)`, not cls!! so inherent PeftModelForCausalLM is no sense
    @classmethod
    def from_pretrained(cls, model, model_id, **kwargs):
        # load the config
        config = LoraConfig.from_pretrained(model_id)

        if getattr(model, "hf_device_map", None) is not None:
            remove_hook_from_submodules(model)

        # here is the hack
        model = cls(model, config)

        # load weights if any
        if os.path.exists(os.path.join(model_id, "adapter_model.bin")):
            filename = os.path.join(model_id, "adapter_model.bin")
        else:
            try:
                filename = hf_hub_download(model_id, "adapter_model.bin")
            except:  # noqa
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {'adapter_model.bin'} is present at {model_id}."
                )

        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        # load the weights into the model
        model = set_peft_model_state_dict(model, adapters_weights)
        if getattr(model, "hf_device_map", None) is not None:
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            no_split_module_classes = model._no_split_modules
            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )
            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    model,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                )
            model = dispatch_model(model, device_map=device_map)
            hook = AlignDevicesHook(io_same_device=True)
            if model.peft_config.peft_type == PeftType.LORA:
                add_hook_to_module(model.base_model.model, hook)
            else:
                remove_hook_from_submodules(model.prompt_encoder)
                add_hook_to_module(model.base_model, hook)
        return model


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-3000")
parser.add_argument("--use_local", type=int, default=1)
args = parser.parse_args()

# max token
MAX_TOKENS = 10000
tokenizer = LlamaTokenizer.from_pretrained(args.model_path, max_length=MAX_TOKENS, truncation=True)

LOAD_8BIT = True
BASE_MODEL = args.model_path
LORA_WEIGHTS = args.lora_path

# fix the path for local checkpoint
lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
print(lora_bin_path)
if not os.path.exists(lora_bin_path) and args.use_local:
    pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
    print(pytorch_bin_path)
    if os.path.exists(pytorch_bin_path):
        os.rename(pytorch_bin_path, lora_bin_path)
        warnings.warn(
            "The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'"
        )
    else:
        assert ('Checkpoint is not Found!')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = SteamGenerationMixin.from_pretrained(
        model, LORA_WEIGHTS, torch_dtype=torch.float16, device_map={"": 0}
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = SteamGenerationMixin.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    print('start torch.compile(model)')
    model = torch.compile(model)

import torch
from langchain.llms.base import LLM
from llama_index import PromptHelper, SimpleDirectoryReader, GPTListIndex
from llama_index import LLMPredictor, GPTSimpleVectorIndex
from llama_index import ServiceContext
from typing import Optional, List, Mapping, Any
import logging
import sys
from transformers import pipeline

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

test_gen_config = GenerationConfig(
    temperature=0.2,
    top_p=0.85,
    top_k=30,
    num_beams=4,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
    max_new_tokens=2500,  # max_length=max_new_tokens+input_sequence
    min_new_tokens=1,  # min_length=min_new_tokens+input_sequence
)


class CustomLLM(LLM):
    model_name = "modified/local"

    # pipeline = pipeline("text-generation",
    #                     model=model,
    #                     tokenizer=tokenizer,
    #                     device=device)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print('# start _call')
        print('prompt: \n', prompt)
        prompt_length = len(prompt)
        # response = self.pipeline(prompt, max_new_tokens=2500)[0]["generated_text"]
        # print(response[prompt_length:])

        print('# break the prompt to tokens')
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # produce output tokens
        # generate_ids = model.generate(input_ids, max_new_tokens=2500, do_sample=True, top_k=30, top_p=0.85,
        #                               temperature=0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1,
        #                               pad_token_id=0)
        print('# generate')
        generate_ids = model.generate(input_ids=input_ids,
                                      generation_config=test_gen_config)

        # decode token to string response
        print('# decode token to string response')
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print('output: \n' + output)
        # slice the prompt
        print('# slice the prompt')
        response = output[prompt_length:]

        # only return newly generated tokens
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


def evaluate(
        input,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        min_new_tokens=1,
        repetition_penalty=2.0,
        **kwargs,
):
    prompt = generate_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens,  # min_length=min_new_tokens+input_sequence
        **kwargs
    )

    print('start text llama-index')
    # TEST
    #
    # set maximum input size
    max_input_size = 2048
    # set number of output tokens
    num_output = 512
    # set maximum chunk overlap
    max_chunk_overlap = 20

    service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=CustomLLM()),
                                                   prompt_helper=PromptHelper(max_input_size, num_output,
                                                                              max_chunk_overlap))

    documents = SimpleDirectoryReader('Chinese-Vicuna/index-docs').load_data()
    print(documents)

    print('start init index')
    index = GPTListIndex.from_documents(documents, service_context=service_context)
    print('end init index done')
    print('start save to disk')
    index.save_to_disk("clash-index.json")
    print('end save to disk')
    # Query and print response
    print('start query')
    response = index.query('what is License of Clash?')
    print('end query')
    # print(response)

    with torch.no_grad():
        # immOutPut = model.generate(input_ids=input_ids, generation_config=generation_config,
        #                            return_dict_in_generate=True, output_scores=False,
        #                            repetition_penalty=float(repetition_penalty), )
        # outputs = tokenizer.batch_decode(immOutPut)
        last_show_text = ''
        for generation_output in model.stream_generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                repetition_penalty=float(repetition_penalty),
        ):
            outputs = tokenizer.batch_decode(generation_output)
            show_text = "\n--------------------------------------------\n".join(
                [output.split("### Response:")[1].strip().replace('�', '') for output in outputs]
            )
            # if show_text== '':
            #     yield last_show_text
            # else:
            yield show_text
            last_show_text = outputs[0].split("### Response:")[1].strip().replace('�', '')


gr.Interface(
    fn=evaluate,
    inputs=[
        gr.components.Textbox(
            lines=2, label="Input", placeholder="Tell me about alpacas."
        ),
        gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
        gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
        gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
        gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="Beams Number"),
        gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=256, label="Max New Tokens"
        ),
        gr.components.Slider(
            minimum=1, maximum=100, step=1, value=1, label="Min New Tokens"
        ),
        gr.components.Slider(
            minimum=0.1, maximum=10.0, step=0.1, value=1.0, label="Repetition Penalty"
        ),
    ],
    outputs=[
        gr.inputs.Textbox(
            lines=15,
            label="Output",
        )
    ],
    title="Chinese-Vicuna 中文小羊驼",
    description="结合 llama-index prompt 搜索优化的 中文小羊驼",
).queue().launch(share=True)
