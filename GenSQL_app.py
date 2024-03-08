import streamlit as st
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer, util 
import os, logging, argparse, time, json, uuid, re, nltk

# nlt. download("punkt")
# nltk. download ("stopwords")
# nltk. data.path.append(" �/data/cache/nItk_data*)
from nitk. corpus import stopwords
from nitk. tokenize import word_tokenize
from prompt_template import text_to_sql_template
from rank_bm25 import BM250kapi
from config import (
	RAG Config,
	logging dir,
	load _embed_model,
	read_data,
	load_vector_index,
	load_bm25_index,
	add_css_styles,
)
import sqlparse


# from sql_metadata import Parsen
from streamlit import session_state as ss 
from streamlit_feedback import streamlit_feedback

def data_preprocess(txt, preprocess_steps=["clean", "stopwords"]):
	"""a function for preprocessing pipeline
	Vector datal
	Args:
		txt (str): input text to clean
		preprocess_ steps (list, optional): preprocessing pipeline. Defaults to [
	Returns:
		str: cleaned text
	"""
	try:
		# 1 clean
		Cleaned_text = re. sub(r" [^0-9a-zA-Z]+", "", txt).lower()
	
		# 2 Remove stop words
		if "stopwords" in preprocess_steps:
			tokens = word_tokenize (cleaned_text)
			stop_words = set(stopwords.words("english"))
			cleaned_text = " ".join([word for word in tokens if word not in stop _words])
		
		result = cleaned_text
	
	except Exception as e:
		# result = txt
		logging.info(f"processing error: {e}")
		raise ValueError (f"processing error: {e}")

	return result

	
def get_bm25filtered_docs(query, basedata, bm25):
	"""
	A function to filter documents based on bm25 algorithm_
	
	Args:
		query (str): user question
		basedata (dataframe): dataframe with original data containing metadata, NLQ's, SQL's 
		bm25 (model): bm25 loaded from index file
	
	Returns:
		dataframe: documents scored from bm25 algorithm
	"""
	try:
		# Get indices of the filtered tables
		tokenized_query = data_preprocess(
			query, preprocess_steps=["clean", "stopwords"]
		). split(" ")
		
		scores = bm25.get_scores(tokenized_query)
		basedata["bm25_score"] = scores
	
	except Exception as e:
		logging. error (f"Error refrieving bm25 based documents: {e}") 
		st.warning(f"Error retrieving documents: {e}", icon="?") 
		raise ValueError(f"Error retrieving bm25 based documents {e}")
	return basedata
	
	
def get_top_similar_doc(query, basedata, model, selected_table, bm25):
	"""
	A function to give top 3 reranked documents after bm25 and vector search
	Args:
		query (str): user question
		basedata (datafrane): dataframe with original data containing metadata, NLO's, SQL's 
		bm25 (algorithm): bm25 model loaded from index file 
		model (model): vector embedding model
		
	Returns:
		dataframe: top 3 reranked documents after om25 and vector search
	"""

	try:
		query_embedding = model.encode(str(query), show_progress_bar=False)
		# query_embedding = model.encode(data_preprocess (query, preprocess_steps=['clean']), show_pi
		
		if selected_table != "None":
			basedata = get_bm25filtered docs (query, basedata, bm25)
			filtdf=(
				basedata [basedata["tables"] = selected_table]
				.sort_values(by="bm25_score", ascending=False)
				.head(rerank)
				)
				
		else:
			basedata = get_bm25filtered_docs (query, basedata, bm25)
			filtdf - basedata.sort_values (by-"bm25_score"s ascending-False) head(menank)
		
		para_embedding = np.array(
		filtdf["embedding_ge_base"].tolist(). dtype=np. float32
		)
		
		similar_res = util.semantic search(
		query_embeddings = query_embedding,
		corpus_embeddings=para_embedding, 
		query_chunk_size=10, 
		corpus_chunk_size=60000, 
		top_k=top_k, 
		score_function=util.cos_sim,
		)
		
		topk_docs = similar_res[0]
		rows_to_filter = [topk_docs[i]["corpus_id*] for i in range(len(topk_docs) )]
		sim_score = [topk_docs[a?r"score"] for 2 in range (Ten(topk_docs))]
		df = filtdf. iloc[rows_to_filter].reset_index(drop=True)
		df ["Score"] = sim_score
		d["Score"] = round(100 * df["Score*]. 2)
		df. rename(columns={"query_summary": "NLQ", "tables": "Tables"}, inplace=True)
		# df = df[~(df[ 'Score' ]=-180.0)]
		
	except Exception as e:
		logging.error(f"Ernonretnieving data. {e}")
		st.warning(f"Error retrieving data: tej" icon=" ")
		raise ValueError (F"Error retrieving data (e}")
		
	return df.head (3)

		
@st.cahce_resournce()
def persist_dataload():
	"""
	persist data load to avoid reloading latency
	"""
	
	vector_embedding = load_vector_index (vectorindex)
	basedata = read_data(dpath)
	basedata ["embedding_bge_base"] = vector_embedding. tolist()
	bm25 = load_bm25_index(bm25index)
	
	return basedata, bm25
	
@st.cache_resource()
def persist_model_load():
	"""
	persist model load to avoid reloading latency
	"""
	
	embed_model_id = RAG_Config.embed_model_id
	model = load _embed_model (embed_model_id)
	
	return model


def write_app_results (query, selected_table, sqlgenerated, display_row):
	"""A function to write all the results generated
	Angs:
	query (str): user questions
	selected_table (str): table selected by usegh if not selected then none 
	sqlgenerated (str): SQL generated from the pipeline
	display_row (json): relevant retrieved data for the given user query which is used as topk_few_shots
	"""

	try:
		data = {
		"user_question:": query,
		"table_selected": selected_table
		"sqlgenerated": sqlgenerated,
		"topk_few shots:": display row,
		}
		
		with open(os-path-join (logdin, "gensal_generation_results json Ma" as file:
			json.dump (data, file) 
			file.write(*\n")

	except Exception as e:
		logging-error(f"Error writing results: fert

		
def reset_cb() :
	"""Reset callback for feedback, etc.
	This can Efused to save interactions.
	"""
	ss. fbk = str(uuid.uuid4()) # use new key to display new feedback prompt.

	
def collect feedback(user_response) :
	"""
	A function to collect the Feedback submitted by user
	"""
	logging. info(f"feedback: (user_response}*)
	with open(os. path. join(logdin, "gensql_generation_results.ison"), "a") as file:
		json.dump ({"user_feedback:": user-pesponse}, file)
		file write("\n")
		

def generate_code (query, selected_table, df):
	"""
	A function to generated code from the inference API after passing the few shot examples
	"""

	logging. info(f"query: {query}")

	start_time = time.time()
	if df.shape[0] <= 1:
		logging.error("not enough samples, choose another table") 
		st.warning( "Not enough samples, choose another table", icon="A*)

	kwargs = {
	"ques1": df["NLQ"].i1oc[0],
	"sq11": df["query_txt"].iloc[0].
	"context1": df ["metadata"].iloc[0],

	"ques2" : df["NLQ"].iloc[1],
	"sq12": df["query_txt"].iloc[1],
	"context2": df["metadata"].iloc[1]

	# 'ques3': df['NLQ'].iloc[2],
	# 'sq13': df['query_txt'].iloc[2],
	# 'context3': df[ metadata Jilo
	}


	ss. resultf, template, contextleng = text_to_sol_template (query, **kwargs)
	# ss.resultf = "resultf"
	# st.write(contextleng)
	st. divider ()
	st. caption(f" :blue[SQL Query Most Relevant to Your Question]")
	display_sql = sqlparse.format(ss.resultf, reindent True, keyword_case-"upper")
	st. code(display_sql, language="sq]")
	ret_time = round((time.time()	- start_time),1)
	# st.write(f"your wait time was (ret_time)s")
	
	logging. info(f"result: logged in (ret_time}sec)
	
	# few shots
	expander = st.expander(":blue[See Relevant Metadata and Queries]")
	# expander.text (f" (template]")
	expander.data_editor(
		df[["NLQ", "metadata", "Tables", "query_txt", "Score"]].head(3),
		column_config=t
		"NLQ" : st.column_config.TextColumn(
		width=200,		help="Natural Language Question relevant to the entered question"
		),
		"metadata": st.column_config.TextColumn(
		"Metadata", width=300, help="Table Description"
		),
		"Tables": st.column_config.TextColumn(
		"Table(s)", width=150, help="Table used in the SQL code"
		) ,
		"query_txt": st.column_config TextColumn("SQL" width=150, help-"SQL Code"),
		"Score": st.column_config.ProgressColumn(
		width=100,
		format="%.2f%%",
		min_value=0, 
		max_value=100,
		help = "similacity scone with the nuestion asked"
		),
		)
		
	#writing results 
	write_app_results(
	query, selected_table, display_sql, df[ ["query_id". "Score"]].head(3).to_json()

	return df


def main():
	# load
	basedata, bm25 = persist_dataload()
	model = persist_model_load()

	# We use uid to generate unique key converted to stra
	if "fbk" net in ss:
		ss.fbk = str(uuid.uuid4())
	# form created
	try:

	with st.container():
		query = st.text_area(
		f":blue[Enter your Natural Language Question to Generate SQL Query: ]"
		"What is average tenure of an account?...",
		)
		
	st .markdown(
	"""
	�style>
		div.stButton � button:first-child f
		background-color: #006FCF; color: white;
		border: 1px solid transparent;
		cursor: pointer;
		display: inline-block; font-weight: 400; max-width: 17.5rem; min-width: 11.25rem; overflow: hidden;
		position: relative; text-align: center;
		text-overflow: ellipsis;
		transition: all .2s ease-in-out;
		transition-property: color, background-color, border-color;
		-webkit-user-select: none;
		-moz-user-select: none;

		moz-user-select:nonel
		user-select: none; vertical-aligne niadle; white-space: nownap:
		padding: 8125ren 1.875neng Font-size: 1rem; line-height: 1.375ren; border-radius: 25rem;
		</style�""",
		unsafe_allow_html=True,
		)
		submitted = st.button ("Submit", on_click=reset_cb)
		
	if submitted:
		ss.selected_table = "None"
		df =  get_top_similar_docs(query,basedata, model, ss.selected_table, bm25)
		
	# add user selection fom tables
		ss.top_tables = df["Tables"].head(5).tolist()
		
		
	except Exception as e:
		logging.error (f"Error in query submission : (el") st.warning(f"Error in query submission (e)", icon="A") raise ValueError(f"Error in query submission(e)")
	
	try:
		with st. form ("main_form"):
		# with st.sidebar:
		if "top_tables" in ss:
			st. radio(
		f":blue[Select a Table to Filter:* options=["None"] + ss. top_tables, key="selected_table", horizontal=True,)
			
			table_select = st. form_submit_button("Generate, on_click=reset_ cbl
	
	if table_select:
	
	logging info(f"table: {ss.selected_table}")
	
	df = get_top_similar docs(
	
	query, basedata, model, ss.selected_table, bm25
	
	generate_code(query, ss.selent)	

	st.write(f":violet[Provide Feedback: Is the generated SQL relevant?]")
	feedback = streamlit_feedback (
	feedback_type="thumbs",
	optional_text_label-"Please mention the right table, Columns, SQL or any additional align="Flex-start", key=ssifibk
	# on_submit=st.write(":violet[Thank You For The Feedback
	if feedback:
		collect_feedback(feedback)
	
	except Exception as e:
		logging.error (f"Error in feedback submission")
		st.warning(f"Error in feedback submissionte)", icon=" A") 
		raise ValueError(f"Erron in feedback submission{e}")
	
	
	if __name__ == "__main__":
	
	# argument parsing
	parser = argparse AngumentParsen(description="GenSOL script")
	parser. add_argument (
				"--logdir"
		type=str, default="./output/**
		help="path to logging and app dump, default will be 'output folder in the current working direct")
		
	args = parser.parse_args()
	logdir = args.logdir
	
	if not os.path.exists("output"):
		os.makedirs("output")
	# print("output folder created")
	logging_dir(os-path.join(logdir, "Gensal_ph1_ SQLGen_app. 10g"))
	
	# set_bg_ hack_ur1()
	add_css_styles()
	config = RAG_Config()
	dpath = config.dpath
	vectorindex = config.vectorindex
	bm25index = vconfig.bm25index
	top_k = vconfig.top_k
	rerank = 20
	main()