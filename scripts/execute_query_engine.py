import argparse
import json
import time

############################
# You can edit you code HERE
# from table_query_engine import initialize_query_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings
from sqlalchemy import create_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_TO_SQL_PROMPT
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import FnComponent
from llama_index.core.llms import ChatResponse
from typing import List
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
    CustomQueryComponent,
)

# put data into sqlite db
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
)
import re
from llama_index.core import SQLDatabase
from sqlalchemy import text
import re
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
import torch


engine = create_engine('sqlite://', echo=False)
model_name = "/project/lt900054-ai2416/train/SuperAI_LLM_FineTune/checkpoint"
df = pd.read_csv("/project/lt900054-ai2416/Data_Test_Table/TBL4-Online-Shopping/TBL4-Online-Shopping-Dataset.csv")
df.to_sql('TableBase', con=engine)

sql_database = SQLDatabase(engine, include_tables=["TableBase"])


def parse_response_to_sql(response: ChatResponse) -> str:
    """Parse response to SQL."""
    response = response.text
    pattern = r"(SELECT.*?;)"

    match = re.findall(pattern, response.replace("\n", " "), re.MULTILINE)
    if match:
      return match[0]

    return response.replace("```sql", "").replace("```", "").strip()

sql_parser_component = FnComponent(fn=parse_response_to_sql)

def sql_parser(query: str):
  with engine.connect() as con:
      rows = con.execute(text(query))
      texts = []
      for row in rows:
          texts.append(str(row))
      return "\n".join(texts)

parser = FnComponent(fn=sql_parser)

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

text2sql_prompt = PromptTemplate(
    """
    [INST] <<SYS>>
    You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด
    <</SYS>>
    {query_str}### (sql extract) {instruction_str} [/INST]
    """
).partial_format(
    instruction_str=table_schema,
    example=sql_parser("SELECT * FROM TableBase LIMIT 5")
)

def tune_component(text_query: str):
    text_query = re.sub("ตอบ.*", "", str(text_query))
    return text_query

def tune_component2(query_str: str):
    text_query = "ตอบ" + re.sub("(.*)ตอบ", "", str(query_str))
    return text_query

tuning = FnComponent(fn=tune_component)
tuning2 = FnComponent(fn=tune_component2)

refine_prompt = PromptTemplate(
    """
    [INST] <<SYS>>
    You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด
    <</SYS>>
    {sql_answer}### {query_str} [/INST]
    """
)

# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )
############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to execute query engine.")
    parser.add_argument(
        "--query-json", type=str, required=True, help="Path to json of quert str."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output.jsonl",
        help="Path to output response.",
    )
    args = parser.parse_args()

    ############################
    # You can edit you code HERE
    # query_engine = initialize_query_engine()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = 'auto',
        torch_dtype = torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # device_map = 'cuda',
        trust_remote_code=True,
    )

    llm = HuggingFaceLLM(
        model=model,
        tokenizer=tokenizer,
        context_window=384,
        max_new_tokens=128,
        tokenizer_kwargs={"max_length": 384},
        generate_kwargs={"temperature": 0.1},
    )
        
    qp = QP(
        modules={
            "input": InputComponent(),
            "tuning": tuning,
            "table_prompt": text2sql_prompt,
            "text2sql_llm": llm,
            "sql_parser": sql_parser_component,
            "parser": parser,
            "refine": refine_prompt,
            "sql2text": llm
        },
        verbose=True,
    )
    qp.add_chain(["input", "tuning", "table_prompt", "text2sql_llm", "sql_parser", "parser"])

    qp.add_link("input", "refine", dest_key="query_str")
    qp.add_link("parser", "refine", dest_key="sql_answer")
    qp.add_chain(["refine", "sql2text"])
    
    class query_engine:
        def __init__(self, query_str: str):
            try:
                self.response = qp.run(query_str=query_str).text
            except Exception as e:
                self.response = "No response"
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
