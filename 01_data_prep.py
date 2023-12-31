# Databricks notebook source
# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will access the [Wayfair Annotation Dataset (WANDS)](https://www.aboutwayfair.com/careers/tech-blog/wayfair-releases-wands-the-largest-and-richest-publicly-available-dataset-for-e-commerce-product-search-relevance), made accessible by [Wayfair](https://www.wayfair.com/) under an MIT License.
# MAGIC
# MAGIC The dataset consists of three file types:
# MAGIC </p>
# MAGIC
# MAGIC * Product - 42,000+ products features on the Wayfair website
# MAGIC * Query - 480 customer queries used for product searches
# MAGIC * Label - 233,000+ product results for the provided queries labeled for relevance
# MAGIC
# MAGIC In the [Annotations Guidelines document](https://github.com/wayfair/WANDS/blob/main/Product%20Search%20Relevance%20Annotation%20Guidelines.pdf) that accompanies the dataset, Wayfair addresses the methods by which queries were labeled.  The three labels assigned to any query result are:
# MAGIC </p>
# MAGIC
# MAGIC * Exact match - this label represents the surfaced product fully matches the search query
# MAGIC * Partial match - this label represents the surfaced product does not fully match the search query
# MAGIC * Irrelevant - this label indicates the product is not relevant to the query
# MAGIC
# MAGIC As explained in the document, there is a bit of subjectivity in assigning these labels but the goal here is not to capture ground truth but instead to capture informed human judgement.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import Required libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn

import os

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./00_config"

# COMMAND ----------

# MAGIC %md
# MAGIC # Download Dataset Files
# MAGIC In this step, we will downlaod the dataset files to a directory accessible within the Databricks workspace:

# COMMAND ----------

# DBTITLE 1,Set Path Variable for Script
os.environ['WANDS_DOWNLOAD_PATH'] = '/dbfs' + config['dbfs_path'] + '/downloads'

# COMMAND ----------

# DBTITLE 1,Download Dataset Files
# MAGIC %sh
# MAGIC
# MAGIC # delete any old copies of temp data
# MAGIC rm -rf $WANDS_DOWNLOAD_PATH
# MAGIC
# MAGIC # make directory for temp files
# MAGIC mkdir -p $WANDS_DOWNLOAD_PATH
# MAGIC # mkdir --help
# MAGIC
# MAGIC # move to temp directory
# MAGIC cd $WANDS_DOWNLOAD_PATH
# MAGIC
# MAGIC # download datasets
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/label.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/product.csv
# MAGIC wget -q https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/query.csv
# MAGIC
# MAGIC # show folder contents
# MAGIC pwd
# MAGIC ls -l

# COMMAND ----------

dbutils.fs.ls('dbfs:/wands/downloads')

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Data to Tables
# MAGIC In this step, we will read data from each of the previously downloaded files and write the data to tables that will make subsequent access easier and faster:

# COMMAND ----------

# MAGIC %sql
# MAGIC -- querying the downloading data
# MAGIC SELECT *
# MAGIC FROM csv. `dbfs:/wands/downloads/product.csv`

# COMMAND ----------

# DBTITLE 1,Process Products
product_schema = StructType([
    StructField('product_id', IntegerType()),
    StructField('product_name', StringType()),
    StructField('product_class', StringType()),
    StructField('category_hierarchy', StringType()),
    StructField('product_description', StringType()),
    StructField('product_features', StringType()),
    StructField('rating_count', FloatType()),
    StructField('average_rating', FloatType()),
    StructField('review_count', FloatType())
])
_ = (
    spark
    .read
    .csv(
        path='dbfs:/wands/downloads/product.csv',
        sep='\t',
        header=True,
        schema=product_schema
    )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save(f"{config['delta_loc']}/bronze/products")
)

# COMMAND ----------

# creating tabel 'products'
products_df = (spark.read.format('delta').load(path=f"{config['delta_loc']}/bronze/products"))
# Write the data to a table.
table_name = "products"
products_df.write.saveAsTable(table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC describe extended products;

# COMMAND ----------

display(spark.table('products'))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM csv. `dbfs:/wands/downloads/query.csv`

# COMMAND ----------

# DBTITLE 1,Process Queries
queries_schema = StructType([
    StructField('query_id', IntegerType()),
    StructField('query', StringType()),
    StructField('query_class', StringType())
])

_ = (
    spark.read.csv(
        path='dbfs:/wands/downloads/query.csv',
        sep='\t',
        header=True,
        schema=queries_schema
    )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save(f"{config['delta_loc']}/bronze/queries")
)

# COMMAND ----------

# creating tabel 'queries'
products_df = (spark.read.format('delta').load(path=f"{config['delta_loc']}/bronze/queries"))
# Write the data to a table.
table_name = "queries"
products_df.write.saveAsTable(table_name)

# COMMAND ----------

display(spark.table('queries'))

# COMMAND ----------

# MAGIC %sql
# MAGIC select query_class, count(query) as number_of_queries
# MAGIC from queries
# MAGIC group by query_class
# MAGIC order by number_of_queries desc;

# COMMAND ----------

# DBTITLE 1,Process Labels
labels_schema = StructType([
    StructField('id', IntegerType()),
    StructField('query_id', IntegerType()),
    StructField('product_id', IntegerType()),
    StructField('label', StringType())
])

_ = (
    spark.read.csv(
        path='dbfs:/wands/downloads/label.csv',
        sep='\t',
        header=True,
        schema=labels_schema
    )
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .save(f"{config['delta_loc']}/bronze/labels")
)

# COMMAND ----------

# creating tabel 'labels'
products_df = (spark.read.format('delta').load(path=f"{config['delta_loc']}/bronze/labels"))
# Write the data to a table.
table_name = "labels"
products_df.write.saveAsTable(table_name)

# COMMAND ----------

display(spark.table('labels'))

# COMMAND ----------

# MAGIC %md 
# MAGIC # Assigning Label Scores
# MAGIC
# MAGIC To prepare the text-based labels assigned to products returned by a query for use in our algorithm, we'll convert the labels to numerical scores based our judgement of how these labels should be weighted:
# MAGIC
# MAGIC **NOTE** [This article](https://medium.com/@nikhilbd/how-to-measure-the-relevance-of-search-engines-18862479ebc) provides a nice discussion of how to approach the scoring of search results for relevance should you wish to explore alternative scoring patterns. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Add Label Score Column to Labels Table
if 'label_score' not in spark.table('labels').columns:
    _ = spark.sql('ALTER TABLE labels ADD COLUMN label_score FLOAT')

# COMMAND ----------

# DBTITLE 1,Assign Label Scores
# MAGIC %sql
# MAGIC  
# MAGIC UPDATE labels
# MAGIC SET label_score = 
# MAGIC   CASE lower(label)
# MAGIC     WHEN 'exact' THEN 1.0
# MAGIC     WHEN 'partial' THEN 0.75
# MAGIC     WHEN 'irrelevant' THEN 0.0
# MAGIC     ELSE NULL
# MAGIC     END;

# COMMAND ----------

display(spark.table('labels'))

# COMMAND ----------


