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
df = pd.read_csv("/project/lt900054-ai2416/Data_Test_Table/Financial/Financial Statements.csv")
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
### Year
Description: Year of financial statement

Data Type: Categorical;

### Company
Description: the symbol of company e.g. AAPL is Apple, MSFT is Microsoft

Data Type: Text;

### Category
Description: The industry of each company.

Data Type: Categorical;

### Market Cap(in B USD)
Description: Market Capacity of each company.

Data Type: Numeric;

### Revenue
Description: Revenue in each company.

Data Type: Numeric;

### Gross Profit
Description: the profit a Company makes after variable production costs but before fixed costs.

Data Type: Numeric;

### Net Income
Description: The amount of accounting profit a company has left over after paying off all its expenses.

Data Type: Numeric;

### Earning Per Share
Description: A company's net income subtracted by preferred dividends and then divided by the average number of common shares outstanding.

Data Type: Numeric;

### EBITDA:
Description: earnings before interest, taxes, depreciation, and amortization.

Data Type: Numeric;

### Share Holder Equity
Description: the amount that the owners of a company have invested in their business.

Data Type: Numeric;

### Cash Flow from Operating
Description: the amount of money a company brings in from its ongoing, regular business activities, such as manufacturing and selling goods or providing a service to customers.

Data Type: Numeric;

### Cash Flow from Investing
Description: Any inflows or outflows of cash from a company's long-term investments.

Data Type: Numeric;

### Cash Flow from Financial Activities
Description: the net amount of funding a company generates in a given time period. Finance activities include the issuance and repayment of equity, payment of dividends, issuance and repayment of debt, and capital lease obligations.

Data Type: Numeric;

### Current Ratio
Description: a company's ability to pay current, or short-term, liabilities (debt and payables) with its current, or short-term, assets (cash, inventory, and receivables).

Data Type: Date;

### Debt/Equity Ratio
Description: used to evaluate a company’s financial leverage and is calculated by dividing a company’s total liabilities by its shareholder equity.

Data Type: Numeric;

### ROE
Description: gauge of a corporation's profitability and how efficiently it generates those profits

Data Type: Numeric;

### ROA
Description: measures the profitability of a company in relation to its total assets

Data Type: Numeric;

### ROI
Description: A ratio that measures the profitability of an investment by comparing the gain or loss to its cost

Data Type: Numeric;

### Net Profit Margin
Description: the percentage of total income you get to keep after all expenses and taxes are paid

Data Type: Numeric;

### Free Cash Flow per Share
Description: measure of a company's financial flexibility that is determined by dividing free cash flow by the total number of shares outstanding

Data Type: Numeric;

### Return on Tangible Equity
Description: the net profit (after interest and tax) as a percentage of the (average) tangible equity or shareholders' funds

Data Type: Numeric;

### Number of Employees
Description: Number of Employee in each company

Data Type: Numeric;

### Inflation Rate(in US)
Description: the rate of increase in prices over a given period of time in US
Data Type: Numeric;
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
        max_new_tokens=32,
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
