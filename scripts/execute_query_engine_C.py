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
df = pd.read_csv("/project/lt900054-ai2416/Data_Test_Table/Spotify/spotify-2023.csv", encoding="latin1")
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
### track_name
Description: Name of the song

Data Type: Text;

### artist(s)_name
Description: Name of the artist(s) of the song

Data Type: Text;

### artist_count
Description: Number of artists contributing to the song

Data Type: Numeric;

### released_year
Description: Year when the song was released

Data Type: Numeric;

### released_month
Description: Month when the song was released

Data Type: Numeric;

### released_day
Description: Day of the month when the song was released

Data Type: Numeric;

### in_spotify_playlists
Description: Number of Spotify playlists the song is included in

Data Type: Numeric;

### in_spotify_charts
Description: Presence and rank of the song on Spotify charts

Data Type: Numeric;

### streams
Description: Total number of streams on Spotify

Data Type: object;

### in_apple_playlists
Description: Number of Apple Music playlists the song is included in

Data Type: Numeric;

### in_apple_charts
Description: Presence and rank of the song on Apple Music charts

Data Type: Numeric;

### in_deezer_playlists
Description: Number of Deezer playlists the song is included in

Data Type: Numeric;

### in_deezer_charts
Description: Presence and rank of the song on Deezer charts

Data Type: Numeric;

### in_shazam_charts
Description: Presence and rank of the song on Shazam charts

Data Type: object;

### bpm
Description: Beats per minute, a measure of song tempo

Data Type: Numeric;

### key
Description: Key of the song

Data Type: Categorical;

### mode
Description: Mode of the song (major or minor)

Data Type: Categorical;

### danceability_%
Description: Percentage indicating how suitable the song is for dancing

Data Type: Numeric;

### valence_%
Description: Positivity of the song's musical content

Data Type: Numeric;

### energy_%
Description: Perceived energy level of the song

Data Type: Numeric;

### acousticness_%
Description: Amount of acoustic sound in the song

Data Type: Numeric;

### instrumentalness_%
Description: Amount of instrumental content in the song

Data Type: Numeric;

### liveness_%
Description: Presence of live performance elements

Data Type: Numeric;

### speechiness_%
Description: Amount of spoken words in the song

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

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
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
