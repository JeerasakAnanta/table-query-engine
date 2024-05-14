import argparse
import json
import time

############################
# You can edit you code HERE
# from table_query_engine import initialize_query_engine
from llama_index.core import SQLDatabase
# from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings
from sqlalchemy import create_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.vllm import Vllm
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(hello)

embed_model = HuggingFaceEmbedding(
    "/project/lt900048-ai24tn/models/BAAI/bge-m3"
)

torch.cuda.empty_cache()

Settings.embed_model = embed_model
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to execute query engine.")
    parser.add_argument(
        "--query-json", type=str, required=True, help="Path to json of quert str."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output123.jsonl",
        help="Path to output response.",
    )
    args = parser.parse_args()

    ############################
    # You can edit you code HERE
    # query_engine = initialize_query_engine()
    #model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/checkpoint/checkpoint3"
    model_name = "/project/lt900048-ai24tn/models/openthaigpt/openthaigpt-1.0.0-13b-chat"
    quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map = 'cuda'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map = 'cuda'
    )

    llms = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer
    )

    # llm = Vllm(
    #     model=model_name,
    #     tensor_parallel_size=4,
    #     max_new_tokens=128,
    #     temperature=0.2,
    # )

    llm = LLM(model="facebook/opt-125m")
    
    df = pd.read_csv("/project/lt900054-ai2416/LANTASONUB/SuperAI_LLM_FineTune/data/TBL5-Customer-Support-Ticket/customer_support_tickets.xlsx")
    _input = 'Columns: '+ ', '.join(df.columns.tolist())
    engine = create_engine('sqlite://', echo=False)
    df.to_sql('TableBase', con=engine)

    sql_database = SQLDatabase(engine)

    query_engines = NLSQLTableQueryEngine(sql_database=sql_database, tables=["TableBase"], llm=llms)

    # print(query_engine.query("How many male customers are there?").response)

    def query_engine(query_str):
        return query_engines.query(query_str)
    ############################

    with open(args.query_json, "r") as f:
        query_json = json.load(f)
    # Reset save_dir
    with open(args.save_dir, "w") as f:
        pass

    for idx, query_str in enumerate(query_json):
        t1 = time.time()
        response = query_engine(query_str).response
        elapsed_time = time.time() - t1
        with open(args.save_dir, "a") as f:
            json.dump(
                {
                    "idx": idx,
                    "query_str": query_str,
                    "response": response,
                    "elapsed_time": elapsed_time,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
