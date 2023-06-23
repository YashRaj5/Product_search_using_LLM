# Databricks notebook source
# MAGIC %md # Introduction
# MAGIC Having demonstrated the basics of assembling a model and supporting data to enable a semantic search, we will now focus on fine-tuning the model. During fine-tuning, the model is fit against a set of data specific to a particular domain, such as our product catalog. The original knowledge accumulated by our model from its pre-training remains intact but is supplemented with information gleaned from the additional data provided. Once the model has been tuned to our satisfaction, it is packaged and persisted just like as before.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %pip install sentence-transformers==2.2.2 langchain==0.0.179 chromadb==0.3.25 typing-inspect==0.8.0 typing_extensions==4.5.0

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from sentence_transformers import SentenceTransformer, util, InputExample, losses, evaluation
import torch
from torch.utils.data import DataLoader
 
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
 
import numpy as np
import pandas as pd
 
import mlflow

# COMMAND ----------

# DBTITLE 1,Get Config settings
# MAGIC %run "./00_config"

# COMMAND ----------

# DBTITLE 1,Get Search Results
# assemble product text relevant to search
search_pd = (
  spark   
    .table('products')
    .selectExpr(
      'product_id',
      'product_name',
      'COALESCE(product_description, product_name) as product_text' # use product description if available, otherwise name
      )
    .join(
      spark
        .table('labels'),
        on='product_id'
      )
    .join(
      spark
        .table('queries'),
        on='query_id'
      )
      .selectExpr('query','product_text','label_score as score')
  ).toPandas()
 
display(search_pd)

# COMMAND ----------

# MAGIC %md We will then download the original model used in the last notebook so that we may convert both the queries and the product text information into embeddings:

# COMMAND ----------

# DBTITLE 1,Download the Embedding Model
# download embeddings model
original_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# DBTITLE 1,Convert Queries & Products to Embeddings
query_embeddings = (
    original_model.encode(
        search_pd['query'].tolist()
    )
)
product_embeddings = (
    original_model.encode(
        search_pd['product_text'].tolist()
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC We can then calculate the cosine similarity between the queries and products associated with them. While we talk about similarity between embeddings as having to do with the distance between two vectors, cosine similarity refers to the angle separating to rays extending from the center of a space to the point identified by the vector (as if it were a coordinate). In a normalized vector space, this angle also captures the degree of similarity between to points:

# COMMAND ----------

# DBTITLE 1,Calculate Cosine Similarity Between Queries and Product
# determine cosine similarity for each query-product pair
original_cos_sim_scores= (
    util.pairwise_cos_sim(
        query_embeddings,
        product_embeddings
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC Averaging these values gives us a sense of how close the queries are to the products in the original embedding space. Please note that cosine similarity ranges from 0.0 to 1.0 with values improving as they approach 1.0:

# COMMAND ----------

# DBTITLE 1,Calculate Avg Cosine Similarity
# average the cosine similarity scores
original_cos_sim_scores = torch.mean(original_cos_sim_scores).item()

# display resutl
print(original_cos_sim_scores)

# COMMAND ----------

# MAGIC %md Examining the correlation between the label scores and the cosine similarity can provide us another measure of the model's performance:

# COMMAND ----------

# DBTITLE 1,Calculate Correlation with Scores
# determine correlation between cosine similarities and relevancy scores
original_corr_coef_score = (
  np.corrcoef(
    original_cos_sim_scores,
    search_pd['score'].values
  )[0][1]
) 
# print results
print(original_corr_coef_score)

# COMMAND ----------

# MAGIC %md
# MAGIC # Fine-Tune the model
# MAGIC With a baseline measurement of the original model's performance in-hand, we can now fine-tune it using our annotated search result data. We will start by restructuring our query results into a list of inputs as required by the model:

# COMMAND ----------

# DBTITLE 1,Restructure Data for Model Input
# define funciton to assemble an input
def create_input(doc1, doc2, score):
    return InputExample(texts=[doc1, doc2], label=score)

# convert each search result into an input
inputs = search_pd.apply(
    lambda s: create_input(s['query'], s['product_text'], s['score']), axis=1).to_list()

# COMMAND ----------

# MAGIC %md 
# MAGIC We will then download a separate copy of our original model so that we may tune it:

# COMMAND ----------

# DBTITLE 1,Download the Embeddings Model
tuned_model = SentenceTransformer('all-MiniLM-L12-v2')

# COMMAND ----------

# MAGIC %md
# MAGIC And we will then tune the model to minimize cosine similarity distances:
# MAGIC
# MAGIC NOTE This step will run faster by scaling up the server used for your single-node cluster.

# COMMAND ----------

# DBTITLE 1,Tune the model
# define instructions for feeding inputs to model
input_dataloader = DataLoader(inputs, shuffle=True, batch_size=16) # feed 16 records at a time to the model
 
# define loss metric to optimize for
loss = losses.CosineSimilarityLoss(tuned_model)
 
# tune the model on the input data
tuned_model.fit(
  train_objectives=[(input_dataloader, loss)],
  epochs=1, # just make 1 pass over data
  warmup_steps=100 # controls how many steps over which learning rate increases to max before descending back to zero
  )

# COMMAND ----------

# MAGIC %md During model fitting, you will notice we are setting the model to perform just one pass (epoch) over the data. We will actually see pretty sizeable improvements from this process, but we may wish to increase that value to get multiple passes if we want to explore getting more. The setting for warmup_steps is just a common one used in this space. Feel free to experiment with other values or take the default.

# COMMAND ----------


