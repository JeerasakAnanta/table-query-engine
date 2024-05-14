import argparse
import json
import time

############################
# You can edit you code HERE
# from table_query_engine import initialize_query_engine
from llama_index.core import SQLDatabase
# from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
from llama_index.core import Settings
from sqlalchemy import create_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


embed_model = HuggingFaceEmbedding(
    "/project/lt900048-ai24tn/models/BAAI/bge-m3"
)

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
        default="./output_lastest.jsonl",
        help="Path to output response.",
    )
    args = parser.parse_args()

    ############################
    # You can edit you code HERE
    # query_engine = initialize_query_engine()
    # model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/checkpoint/checkpoint7"
    # model_name = "/project/lt900048-ai24tn/models/openthaigpt/openthaigpt-1.0.0-13b-chat"
    # model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/checkpoint/checkpoint4"
    model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/Checkpoint/checkpoint5"
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map = 'auto'
    )
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map = 'auto'
    )

    llms = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer
    )

    df = pd.read_csv("/project/lt900054-ai2416/400850-Tong/Dataset/openthaigpt-exercise-ungraded/TBL4-Online-Shopping-Dataset.csv")
   
    engine = create_engine('sqlite://', echo=False)
    df.to_sql('TableBase', con=engine)
    table_schema = """
    ### CustomerID
    Description: Unique identifier for each customer.

    Data Type: Numeric;

    ### Gender
    Description: Gender of the customer (e.g., Male, Female).

    Data Type: Categorical;

    ### Location
    Description: Location or address information of the customer.

    Data Type: Text;
    ### Tenure_Months
    Description: Number of months the customer has been associated with the platform.
    Data Type: Numeric;

    ### Transaction_ID
    Description: Unique identifier for each transaction.

    Data Type: Numeric;

    ### Transaction_Date
    Description: Date of the transaction.

    Data Type: Date;

    ### Product_SKU
    Description: Stock Keeping Unit (SKU) identifier for the product.

    Data Type: Text;

    ### Product_Description
    Description: Description of the product.

    Data Type: Text;

    ### Product_Category:
    Description: Category to which the product belongs.

    Data Type: Categorical;

    ### Quantity
    Description: Quantity of the product purchased in the transaction.

    Data Type: Numeric;

    ### Avg_Price
    Description: Average price of the product.

    Data Type: Numeric;

    ### Total_Price
    Description: Total price of the product exclude delivery charges.

    Data Type: Numeric;

    ### Delivery_Charges
    Description: Charges associated with the delivery of the product.

    Data Type: Numeric;

    ### Date
    Description: Date of the transaction (potentially redundant with Transaction_Date).

    Data Type: Date;

    ### Month
    Description: Month of the transaction.

    Data Type: Categorical;

    """
    sql_database = SQLDatabase(engine)
    text2sql_prompt = PromptTemplate(
        "Below is an instruction that describes a task, paired with an input that provides further context. "  # noqa: E501
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction_str}\n\n### Input:\n{query_str}\n\n### Response:"  # noqa: E501
    ).partial_format(
    instruction_str=table_schema,
)

    query_engines = NLSQLTableQueryEngine(sql_database=sql_database, tables=["TableBase"], llm=llms, text_to_sql_prompt=text2sql_prompt)

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
