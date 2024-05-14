from llama_index.core import SQLDatabase
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings
from sqlalchemy import create_engine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
import pandas as pd
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
model_name = "/project/lt900048-ai24tn/models/openthaigpt/openthaigpt-1.0.0-13b-chat"

# Settings.embed_model = 
embed_model = HuggingFaceEmbedding(
    "/project/lt900048-ai24tn/models/BAAI/bge-m3"
)

Settings.embed_model = embed_model

class SqlQueryPipeline:
    def __init__(self, df_path, table_name, llms, embed_model):

        self.df = pd.read_csv(df_path)
        self.engine = create_engine('sqlite://', echo=False)
        self.df.to_sql('TableBase', con=self.engine)
        

        # self.connection_string = "sqlite:///Database/TBL4-Online-Shopping-Dataset.db"
        print("connection string: ")
        # self.engine = create_engine(self.connection_string)
        print("engine: ")
        self.db = SQLDatabase(self.engine)
        print("db: ")
        self.llms = llms
        self.embed_model = embed_model
        self.table_name = table_name
        self.query_engine = NLSQLTableQueryEngine(
                                sql_database=self.db, tables=[table_name], llm=self.llms
                            )
        print("Complete query engine: ")
    
    def __call__(self, query):
        return self.query_engine.query(query)


###########################################################################
#
#                            Call Query Pipeline
#
###########################################################################

quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_quant_type="nf8",
        bnb_8bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)

llms = HuggingFaceLLM(
    model=model,
    tokenizer=tokenizer
)

pipeline = SqlQueryPipeline('Dataset/openthaigpt-exercise-ungraded/TBL4-Online-Shopping-Dataset.csv', 'TableBase', llms, embed_model)

print(pipeline("มี transitions เท่าไหร่ที่ถูกสร้างในเดือนเมษา"))
print(pipeline("มีผู้ชายกี่คนที่อาศัยอยู่ในนิวยอร์ก"))

# embeddings = embed_model.get_text_embedding("Hello World!")
# print(len(embeddings))

print("complete..........")