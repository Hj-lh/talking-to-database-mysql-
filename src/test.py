import mysql.connector
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_db_connection():
    connection = mysql.connector.connect(
        host="",
        user="",
        password="",
        database=""
    )
    return connection

def get_db_schema(connection):
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    schema = {}
    for table in tables:
        cursor.execute(f"DESCRIBE {table[0]}")
        schema[table[0]] = cursor.fetchall()
    cursor.close()
    return schema

# Load model and tokenizer
model_name = "Qwen/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype ="auto", device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimize CPU usage if using CPU
if device == torch.device("cpu"):
    torch.set_num_threads(12)  # Adjust this based on the number of available CPU cores

def generate_sql_query(user_question, schema):
    
    
    prompt = f"Based on the schema {schema}, write an SQL query for the question: {user_question}\nSQL:"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=80, num_return_sequences=1)
    query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the SQL query from the model's output
    if "SQL:" in query:
        query = query.split("SQL:")[1].strip()
    print("----------------------------------------------------------")
    print(query)
    print("----------------------------------------------------------")

    return query

def execute_query(connection, query):
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

def generate_response(query_result,user_question):
    # input_text = f"""Based on the following data: {query_result},and user question: {user_question}, generate an answer response."""
    input_text = f"""
        based on user question: {user_question} and the answer for it: {query_result} Generate response: 
        
        """
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024).to(device)
    outputs = model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    user_question = "what subjects are there"
    connection = get_db_connection()
    schema = get_db_schema(connection)

    sql_query = generate_sql_query(user_question, schema)
    print(f"Generated SQL Query: {sql_query}")

    query_result = execute_query(connection, sql_query)
    print(f"Query Result: {query_result}")

    response = generate_response(query_result,user_question)
    print(f"Response: {response}")

    connection.close()

if __name__ == "__main__":
    main()
