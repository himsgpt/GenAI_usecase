import requests, json, config, re, os, logging, sqlparse
from config import RAG_Config, logging_dir

# show model info
# print(requests request ('GET", FI(API_URL;/info) verify-False) content, decode ("utf-8"), -n\n*)
config = RAG_Config()
API_URL = config.API_URL


def query(payload):
    """send	and recieve request	to LLM API'S_
    Args:
            payload (type): input load for sending to API

    Returns:
            str:	output from the API
    """
    #
    try:
        #
        data = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        response = requests.request(
            "POST", f"{API_URL}", headers=headers, verify=False, json=payload
        )
        output = json.loads(response.content.decode("utf-8"))
        # print (data)

    except Exception as e:
        logging.error(f"Error accessing Genie API: {e}")
        raise ValueError(f"Error accessing Genie API{e}")

    return output


def extract_schema_definition(schemal, schema2, p):
    try:
        if p == "2shot":
            temp = schemal.split("\n") + schema2.split("\n")
            unique_list = []
            for item in temp:
                if item not in unique_list:
                    unique_list.append(item)

            schema_comb = "\n".join(unique_list)

        else:
            schema_comb = schemal

    except Exception as e:
        logging.error(f"unable to combine schema")
        raise ValueError("unable to combine schema")
    
	return schema_comb
    
def text_to_sql_template(user_question, **kwargs):
	"""
	template for generating response_
	Args:
		user-question (str): input user question
		**kwangs: keyword argument containing few shot examples to be fed to temolate
        
    Returns:
		str: output from the inference
	"""
    
    try:
        text_2_sql_template_2s = f"""<s>[INST] <<SYS>>
You are an expert SQL coder. Your task is to generate a syntactically correct SOL code from the given English question. Pay attention to use the below instructions in bullet p
* Please generate only 1 simplified SQL code and wrap your code between [SQL] and [/SQL] tags. Do not generate any additional note.
* Only generate SOL code for the specific question asked by the user. The examples may contain additional filter but do not include additional filters or conditions unless exp
* Pay attention to the column names mentioned in the CONTEXT section. Do not generate imaginary columns.
* Use the provided examples as a reference for the expected format. Strictly do not repeat the example in your answers.
* If you do not have sufficient context or knowledge from the examples to write the SOL code, directly write "I need more context to generate code'.
* Pay attention to strictly not hallucinate and do not generate false information at any time. Be careful to not create imaginary columns that do not exist.
<</SYS>>

[QUESTION]: {kwargs['ques1']}
[CONTEXT]: Let's think step by step to write sql code from given question by using the below columns description and table description.
{kwargs['context1']}
[SQL]:
{kwargs['sql1'].strip()}
[/SQL]

[QUESTION]: {kwargs['ques2']}
[CONTEXT]: Let's think step by step to write sql code from given question by using the below columns description and table description.
{kwargs['context2']}
[SQL]:
{kwargs['sql2'].strip()}
[/SQL]

[QUESTION]: {user_question}
[SOL]:
[/INST]
"""

        text_2_sql_template_1s = f"""<s>[INST] <<SYS>>
You are an expert SQL coder. Your task is to generate a syntactically correct SOL code from the given English question. Pay attention to use the below instructions in bullet p
* Please generate only 1 simplified SQL code and wrap your code between [SQL] and [/SQL] tags. Do not generate any additional note.
* Only generate SOL code for the specific question asked by the user. The examples may contain additional filter but do not include additional filters or conditions unless exp
* Pay attention to the column names mentioned in the CONTEXT section. Do not generate imaginary columns.
* Use the provided examples as a reference for the expected format. Strictly do not repeat the example in your answers.
* If you do not have sufficient context or knowledge from the examples to write the SOL code, directly write "I need more context to generate code'.
* Pay attention to strictly not hallucinate and do not generate false information at any time. Be careful to not create imaginary columns that do not exist.
<</SYS>>

[QUESTION]: {kwargs['ques1']}
[CONTEXT]: Let's think step by step to write sql code from given question by using the below columns description and table description.
{kwargs['context1']}
[SQL]:
{kwargs['sql1'].strip()}
[/SQL]

[QUESTION]: {user_question}
[SOL]:
[/INST]
"""
        
        if len(text_2_sql_template_2s) < 7500:
            text_2_sql_template = text_2_sql_template_2s
            schema_definition = extract_schema_definition(
                kwargs ["context1"], kwargs ["context2"], p="2shot")
            
        else:
            text_2_sql_template = text_2_sql_template_1s
            schema_definition = extract_schema_definition(
                kwargs ["contextl"], kwargs["context2"], P="1shot")

        
        ans = query({
            "prompt": text_2_sql_template,
            "max_tokens": 200,
            "temperature": 0.1,
            "top_p": 0.1,
            "top_k*: 40,
            "model": "codeLlama-34b-instr-40k",
            })
        
        result = ans["choices"][0]["text"].strip()


        # post processing
        try:
            pattern = r"(\[?SQL]?)(.*?)(\[?/SQL]?)"
            res = re.findall(pattern, result, re.DOTALL) [9]
            res =  res[0] + res[1]

        except:
             res = result

    except Exception as e:
         logging.error(f"Generation error")


    return res, schema_definition, len(text_2_sql_template) 
