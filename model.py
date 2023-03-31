import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, GPTSimpleVectorIndex, ServiceContext
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig, LlamaTokenizer
from transformers import pipeline
from typing import Optional, List, Mapping, Any


class CustomLLM(LLM):
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    model_name = 'author/model_name'
    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.85,
        top_k=2,
        num_beams=4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_new_tokens=2500,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=1,  # min_length=min_new_tokens+input_sequence
    )
    model : Optional[LlamaTokenizer]
    tokenizer : Optional[LlamaTokenizer]
    device = 'cuda'
    # pipeline = pipeline("text-generation",
    #                     model=model,
    #                     tokenizer=tokenizer,
    #                     device=device)

    def __init__(self, mod, token, gen_config, device, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.generation_config = gen_config
        self.model = mod
        self.tokenizer = token
        self.device = device

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.doCall(self, prompt, stop)

    def doCall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print('# start _call')
        print('prompt: \n', prompt)
        prompt_length = len(prompt)
        # response = self.pipeline(prompt, max_new_tokens=2500)[0]["generated_text"]
        # print(response[prompt_length:])

        print('# break the prompt to tokens')
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # produce output tokens
        # generate_ids = model.generate(input_ids, max_new_tokens=2500, do_sample=True, top_k=30, top_p=0.85,
        #                               temperature=0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1,
        #                               pad_token_id=0)
        print('# generate')
        generate_ids = self.model.generate(input_ids=input_ids,
                                           generation_config=self.generation_config)

        # decode token to string response
        print('# decode token to string response')
        output = \
            self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print('output: \n' + output)
        # slice the prompt
        print('# slice the output, only newly generated token stay')
        response = output[prompt_length:]

        return response

    def doPipelineCall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"
