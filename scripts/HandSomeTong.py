import argparse
import json
import time

############################
# You can edit you code HERE
# from table_query_engine import initialize_query_engine

from typing import List
import torch
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.llms import ChatResponse
from llama_index.core import SQLDatabase, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.vllm import Vllm
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    text,
)
import re
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import bitsandbytes as bnb

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.empty_cache()

def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.text
    pattern = r"(SELECT.*?;)"

    match = re.findall(pattern, response.replace("\n", " "), re.MULTILINE)
    if match:
        return match[0]

    return response.replace("```sql", "").replace("```", "").strip()

def sql_parser(query: str):
    df = pd.read_csv("/project/lt900054-ai2416/400850-Tong/Dataset/openthaigpt-exercise-ungraded/TBL4-Online-Shopping-Dataset.csv")
    engine = create_engine('sqlite://', echo=False)
    df.to_sql('TableBase', con=engine)

    with engine.connect() as con:
        rows = con.execute(text(query))
        texts = []
        for row in rows:
            texts.append(str(row))
        return "\n".join(texts)

# def tune_component(text: str):
#     return text.split("\n")[0]

def tune_component(text_query: str):
    text_query = re.sub("ตอบ.*", "", str(text_query))
    return text_query


def create_query_pipeline(llm, text_2_text, text2sql_prompt, refine_prompt) -> QP:
    qp = QP(
        modules={
            "input": InputComponent(),
            # "tuning": FnComponent(fn=tune_component),
            # "tune_text": text_2_text,
            # "tune_parser": FnComponent(fn=tune_component),
            # "llm1": llm,
            "table_prompt": text2sql_prompt,
            "text2sql_llm": llm,
            "sql_parser": FnComponent(fn=parse_response_to_sql),
            "parser": FnComponent(fn=sql_parser),
            "refine": refine_prompt,
            "sql2text": llm
        },
        verbose=True,
    )

    qp.add_chain(["input", "table_prompt", "text2sql_llm", "sql_parser", "parser"])
    # qp.add_link("llm1", "table_prompt", dest_key="query_str")
    # qp.add_chain(["table_prompt", "text2sql_llm", "sql_parser", "parser"])
    qp.add_link("parser", "refine", dest_key="sql_answer")
    qp.add_link("input", "refine", dest_key="query_str")
    qp.add_chain(["refine", "sql2text"])

    return qp

###################################################################################################
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

# text2sql_prompt = PromptTemplate(
#     """
#     You are advanced SQL code generator from query.
#     The table name is 'TableBase'

#     If the instruction say please answer or must answer in format ..... YOU GOING TO IGNORE THEM!

#     ## The five first rows of TableBase.
#     {example}

#     ## Consider these fields
#     {instruction_str}

#     ## Follow these instructions:
#       1. The final line of code should be a SQL code that can be run in SQL query engine.
#       2. The total number of customers must use DISTINCT function to count.
#       3. Available table is 'TableBase'

#     Query: {query_str}
#     One-line Expression:
#     """
# ).partial_format(
#     instruction_str=table_schema,
#     example=sql_parser("SELECT * FROM TableBase LIMIT 5")
# )

text2sql_prompt = PromptTemplate(
    """
   You are advanced SQL code generator from query.
    The table name is TableBase

    ## Follow these instructions:
      
The final line of code should be a SQL code that can be run in SQL query engine.
The total number of customers must use DISTINCT function to count.
Don't use any non-existing column.

    ## Consider these fields
    {instruction_str}

    Note: TableBase in only available table.
    Note: Don't use other column than the one in the table.

    Query: {query_str}
    One-line Expression:
    """
).partial_format(
    instruction_str=table_schema,
    example=sql_parser("SELECT * FROM TableBase LIMIT 5")
)

text_2_text = PromptTemplate(
    """
    Your are expert linguistics.
    You will extract the important part of the input

      Ex. 1
      Input:  How many female customers please answer in json format 'total': ....
      Output: How many female customers

      Ex. 2
      Input:  How many customers please answer in array format.
      Output: How many customers


    Input: {query_str}
    Answer:
    """
)

refine_prompt = PromptTemplate(
    """
    Re-format the sql output based on instruction.
    Ex.
       ins: Answer in array format
       inf: 1000, 2000 and 5000
       ans: [1000, 2000, 5000]


    ins: {query_str}
    inf: {sql_answer}
    ans:
    """
)


# text2sql_prompt = PromptTemplate("""
#     instruction_str: {instruction_str}
#     Query: {query_str}
#     One-line Expression:
# """)
# text_2_text = PromptTemplate("")
# refine_prompt = PromptTemplate("")

############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to execute query engine.")
    parser.add_argument(
        "--query-json", type=str, required=True, help="Path to json of quert str."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output_model3.jsonl",
        help="Path to output response.",
    )
    args = parser.parse_args()

    ############################
    # You can edit you code HERE
    # query_engine = initialize_query_engine()
    # model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/checkpoint/checkpoint4"
    # model_name = "/project/lt900048-ai24tn/models/openthaigpt/openthaigpt-1.0.0-13b-chat"
    model_name = "/project/lt900048-ai24tn/models/SeaLLMs/SeaLLM-7B-v2.5"
    # model_name = "/project/lt900048-ai24tn/models/pythainlp/wangchanglm-7.5B-sft-enth"
    # model_name = "/project/lt900048-ai24tn/models/openthaigpt/openthai-mistral-21000"
    # model_name = "/project/lt900048-ai24tn/models/airesearch/WangchanLion7B/"
    # model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/checkpoint/checkpoint_openthai_13b_olddata"
    # model_name = "/project/lt900054-ai2416/ILOVELANTA/SuperAI_LLM_FineTune/Checkpoint/checkpoint5"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = 'cuda',
        quantization_config = bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map = 'cuda',
        trust_remote_code=True,
    )

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=512,
        max_new_tokens=100,
        tokenizer_kwargs={"max_length": 100},
        generate_kwargs={"temperature": 0.2, 'num_beams': 10,},
    )

    # llm = Vllm(
    #     model=model_name,
    #     tensor_parallel_size=4,
    #     max_new_tokens=128,
    #     temperature=0.1,
    #     vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
    # )

    qp = create_query_pipeline(llm, text_2_text, text2sql_prompt, refine_prompt)

    
    def query_engine(query_str):
        return qp.run(query_str=query_str)
    

    ############################

    with open(args.query_json, "r") as f:
        query_json = json.load(f)
    # Reset save_dir
    with open(args.save_dir, "w") as f:
        pass

    for idx, query_str in enumerate(query_json):
        t1 = time.time()
        try:
            response = query_engine(query_str).text #response
        except Exception as e:
            response = str(e)
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
