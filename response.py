import torch
from langchain.llms.base import LLM
from llama_index import SimpleDirectoryReader, GPTListIndex, PromptHelper
from llama_index import LLMPredictor, GPTSimpleVectorIndex,ServiceContext
from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import pipeline
from typing import Optional, List, Mapping, Any

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)




class CustomLLM(LLM):

    model_name = "facebook/opt-iml-max-30b"
    pipeline = pipeline("text-generation", model=model_name, device="cuda:0",
                        model_kwargs={"torch_dtype": torch.bfloat16})

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt_length = len(prompt)
        # response = self.pipeline(prompt, max_new_tokens=num_output)[0]["generated_text"]

        # BELlE torch 运行
        ckpt = ''
        # set device (cuda or cpu)
        device = torch.device('cuda')
        # load model
        model = LlamaForCausalLM.from_pretrained(ckpt, device_map='auto', low_cpu_mem_usage=True)
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt)

        # prompt = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "

        # break the prompt to tokens
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # produce output tokens
        generate_ids = model.generate(input_ids, max_new_tokens=500, do_sample=True, top_k=30, top_p=0.85,
                                      temperature=0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1,
                                      pad_token_id=0)

        # decode token to string response
        output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # slice off the prompt
        response = output[len(prompt):]

        # only return newly generated tokens
        return response[prompt_length:]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


# define our LLM
llm_predictor = LLMPredictor(llm=CustomLLM())

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# Load the your data
# documents = SimpleDirectoryReader('./data').load_data()
# index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

index = GPTSimpleVectorIndex.load_from_disk('llama-index.json',llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# Query and print response
response = index.query("<query_text>")
print(response)
