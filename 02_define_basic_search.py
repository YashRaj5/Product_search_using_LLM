# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC With our data in place, we will now take an off-the-shelf model and apply it to perform product search. A key part of this work is the introduction of a vector database that our model will use during inference to rapidly search the product catalog.
# MAGIC
# MAGIC To understand the vector database, you first need to understand *embeddings*. An embedding is an array of numbers that indicate the degree to which a unit of text aligns with clusters of words frequently found together in a set of documents. The exact details as to how these numbers are estimated isn't terribly important here.  What is important is to understand that the mathematical distance between two embeddings generated through the same model tells us something about the similarity of two documents.  When we perform a search, the user's search phrase is used to generate an embedding and it's compared to the pre-existing embeddings associated with the products in our catalog to determine which ones the search is closest to.  Those closest become the results of our search.
# MAGIC
# MAGIC To facilitate the fast retrieval of items using embedding similarities, we need a specialized database capable of not only storing embeddings but enabling a rapid search against numerical arrays. The class of data stores that addresses these needs are called vector stores, and one of the most popular of these is a lightweight, file-system based, open source store called [Chroma](https://www.trychroma.com/).  
# MAGIC
# MAGIC In this notebook, we will download a pre-trained model, convert our product text to embeddings using this model, store our embeddings in a Chroma database, and then package the model and the database for later deployment behind a REST API.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install sentence-transformers==2.2.2 langchain==0.0.179 chromadb==0.3.25 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# MAGIC %pip install urllib3==1.26.6  pyopenssl certifi requests

# COMMAND ----------

pip list

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sentence_transformers import SentenceTransformer
 
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
 
import mlflow
 
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_config"

# COMMAND ----------

# DBTITLE 1,Get Product Text to Search
# assemble product text relevant to search
product_text_pd = (
  spark
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
  ).toPandas()
 
display(product_text_pd)

# COMMAND ----------

# MAGIC %md ## Convert Product Info into Embeddings
# MAGIC
# MAGIC We will now convert our product text into embeddings.  The instructions for converting text into an embedding is captured in a language model.  The [*all-MiniLM-L12-v2* model](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) is a *mini language model* (in contrast to a large language model) which has been trained on a large, well-rounded corpus of input text for good, balanced performance in a variety of document search scenarios.  The benefit of the *mini* language model as compared to a *large* language is that the *mini* model generates a more succinct embedding structure that facilitates faster search and lower overall resource utilization.  Given the limited breadth of the content in a product catalog, this is the best option of our needs:
# MAGIC

# COMMAND ----------

 import requests
 import os
 os.environ['REQUESTS_CA_BUNDLE'] = '/path/to/certifi/cacert.pem'

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# MAGIC %md
# MAGIC To use our model with our vector store, we need to wrap it as a LangChain HuggingFaceEmbeddings object. We could have had that object download the model for us, skipping the previous step, but if we had done that, future references to the model would trigger additional downloads. By downloading it, saving it to a local path, and then having the LangChain object read it from that path, we are bypassing unnecessary future downloads:

# COMMAND ----------

# DBTITLE 1,Load Model as HuggingFaceEmbeddings Object
# encoder path
embedding_model_path = f"/dbfs{config['dbfs_path']}/embedding_model"
 
# make sure path is clear
dbutils.fs.rm(embedding_model_path.replace('/dbfs','dbfs:'), recurse=True)
 
# reload model using langchain wrapper
original_model.save(embedding_model_path)
embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_path)

# COMMAND ----------

display(dbutils.fs.ls(f"dbfs:{config['dbfs_path']}/embedding_model"))

# COMMAND ----------

# MAGIC %md
# MAGIC Using our newly downloaded model, we can now generate embeddings. We'll persist these to the Chroma vector database, a database that will allow us to retrieve vector data efficiently in future steps:

# COMMAND ----------

# DBTITLE 1,Reset Chroma File Store
# chromadb path
chromadb_path = f"/dbfs{config['dbfs_path']}/chromadb"
 
# make sure chromadb path is clear
dbutils.fs.rm(chromadb_path.replace('/dbfs','dbfs:'), recurse=True)

# COMMAND ----------

# DBTITLE 1,Load Product info for use with Encoder
# assemble product documents in required format (id, text)
documents = (
  DataFrameLoader(
    product_text_pd,
    page_content_column='product_text'
    )
    .load()
  )

# COMMAND ----------

documents
