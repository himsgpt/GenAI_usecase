fron typing import Dict
import logging, os, pandas as pd, pickle
from sentence_transformers import SentenceTransformer 
import streamlit as st

@dataclass
Class RAG_Config:
	API_URL: str - field(
	default="https://goldbigdatalabs.aexp.com/genie/ip:port/v1/completions", metadata=["help":"*Inference API end points"),
	embed_model_id: str = field(
	default=./data/bge-base-en-v1.5
	metadata=["help": "embedding model id",
	dpath: str = field(
	default=*./data/phase2_55k_v1_masked.parquet.
	metadata=("help": "input file name w/ path (esv and parquet supported) ",
	vectorindex: str = fieldi
	default=*./data/phase2_55k_v1_bge-base.pkl,
	metadata= ("help": "embedding pickle file created on query_summary"},
	bm25index: str = Field(
	default="./data/phase2_55k_v1_bm25index.pkl",
	metadata=("help": "bm25 indexed on metadata"),
	top_k: int - field(
	defauat=3,
	metadata=("help": "integral value to determine number of results to retrieve"),
	)
	
def read_data(dpath):
	"""
	function to read input data_
	Args:
		dpath (str): input data path with metadata, sql code and summary
	Raises:
		FileNotFoundError: if the file is not found
	Returns:
		dataframe: dataframe containing the data
	"""

# input file
	if not os.path.exists(dpath):
		logging. error (f*Data path does not exist: (dpath}") raise FileNotFoundErron(f"Invalid data path (dpathy")
	
	try:
		if dpath.lower().endswith("csv"):
			df = pd.read_csv(apath)
			logging info(f"input data read {dpath)")
			
		elif dpath. lower ().endswith("parquet"):
			df = pd. read_ parquet (dpath, engine="pyarrow")
			logging info(f"input data read (dpath}")
			
		else:
			logging. error (f"Data format not supported. Currently only csy, parquet file supported (dpath}")
			raise ValueError(f"Data format not supported. Currently only SM, parquet file supported, check logs")

	except Exception as e:
		logging.error (f"Error reading data: (e}*)
		raise ValueError ("Error reading data, check logs")

	return df
	
	
# logging
def logging dir(logpath) :
	"""a function to call for logging
	Args:
	logpath (str): directory where logging is to be saved. Default is output directory
	"""
	if not os.path.exists ("output" ):
		os.makedirs("output")
		# logging. info("output directory created")
	
	logging. basicConfig(
		Filename=logpath, level=logging. INFO,
		format-"%(asctime)s - %(levelname)s: - %(message) 5" ,
		
	)
	
def load_embed_model (embed_model_id) :
	"""
	a function to load the vector embedding models
	Args:
	embed_model_id (str): embedding model with path
	Returns:
	model: vector model
	"""
	if not os path.exists(embed_model_id):
		logging error (f"embedding model path does not exist: {embed_model_id}") 
		raise FileNotFoundError (f"Invalid embedding model path(embed _model_id) ")
		
	try:
		model = SentenceTransformer (
		embed_model_id
		# device=' cuda'
		)
		logging.info(f"embedding model id loaded {os.path.basename(embed_model_id)}")
	
	except Exception as e:
		logging, error (f"Error reading embeddine model



def load vector_index(infile) :
	"""_a function to load vector index created from vecton embedding model on the input data
	Args:
	infile (str): vector index file with location
	Retunns :
	type: index
	"""
	if not os.path.exists(infile):
		logging. error (f"embedding pickle file not found: (infile)") 
		raise FileNotFoundError (f"embedding pickle file not found(infile)")
	
	try:
		with open(infile, "rb") as fIn:
			stored _data = pickle.load(FIn)
			# stored_sentences = stored _data[ sentences']
			vector_embedding = stored_data[ "embeddings"]

		logging. info("dimension of vector embeddings pickle is {len(vector_embedding)} rows, {len(vector_embedding[0])} dim loaded")
		
	except Exception as e:
		logging. error (f"Error reading embedding pickle file: {e}")
		raise ValueError ("Error reading embedding pickle file, check logs")
	return vector _embedding
	
	
def load_bm25_index(infile):
	"""	_a function to load bm25 index created from bm25 rank algorithm on the input data
	Args:
	infile (str): bm25 index the with location
	
	Returns:
	-type_: index
	"""
	if not os.path.exists(infile):
		logging.error(f"bm25 pickle file not found: {infile}")
		raise FileNotFoundError(f"bm25 pickle file not foundf {infile}")
	
	try:
		with open(infile, "rb") as +In:
		bm25 = pickle.load(fIn)
		logging. info(f"bm25 index of size (bm25.corpus_size} docs loaded from (infile)")
	except Exception as e:
		logging. error (f"Error reading bm25 pickle file: (e?") raise ValueError ("Error reading bm25 pickle file")
	
	return bm25


def add_css_styles():
"""a function to add css styles for defining amex template"""

	try:
	# 1
	st.set_page_config(layout="wide")
	# 2
	st.markdown (
	# st. title("Streamlit Titie")
	< DOCTYPE html>
	<html lang="en"=
	‹head›
	‹title›AILabs | GenSQl</title>
	‹meta charset="utf-8">
	‹meta name="viewport" content="wiath-device-width, initial-scale-1.0, maxinum-scale=1.0, user-scalable=no"› «script src="./jquery-3.7.1.min-js*></script>
	<script src="./marked.min.js"></script>
	<link type="text/css" rel="stylesheet* href=*https://www.aexp-static.com/cdaas/one/statics/@americanexpress/dls/6.25.5/package/dist/6.25.5/style
