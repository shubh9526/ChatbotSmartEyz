from flask import Flask, request, jsonify
import json
import urllib.parse
from sqlalchemy import create_engine
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
import ast
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enables CORS for all routes


# SQLserver Database Configuration
server = "103.91.187.244,1433;"
database = "SmartEyzAI"
driver = "ODBC Driver 17 for SQL Server"
username = "smarteyzchatbot"             # Your SQL login
password = "SmartEyz@2025"      # Use the correct password

params = urllib.parse.quote_plus(
    f"DRIVER={{{driver}}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"TrustServerCertificate=yes;"
)

connection_string = f"mssql+pyodbc:///?odbc_connect={params}"


def create_db_engine():
    """
    Creates and returns a database engine using SQLAlchemy.
    """

    # Return a database engine by calling `create_engine` with a connection string.
    return create_engine(connection_string)


def initialize_database():
    """
    Initializes the SQL database connection.
    """
    # Call the `create_db_engine` function to create a database engine and assign it to `engine`.
    engine = create_db_engine()
    # Return an initialized `SQLDatabase` object using the created engine
    return SQLDatabase(engine)
#
#
# # Call the `initialize_database` function to set up the database connection and store the result in `db`.
db = initialize_database()
#
#
# # Initialize OpenAI Chat Model
def initialize_chat_model():
    """
    Initializes and returns the OpenAI Chat model.
    """
    return AzureChatOpenAI(
        api_key="4AzMtLTJhRrfhqzJ0Y8YZXZ81yG5bKa5edvBwz3uMARCLwqvbXu4JQQJ99BCACYeBjFXJ3w3AAABACOG2GKH",
        # api_key="XXXXXXXXXXX",
        azure_endpoint='https://smarteyzai.openai.azure.com/',
        api_version='2024-09-01-preview',
        temperature=0,  # Lower temperature makes responses more predictable
        model="gpt-4o-mini"
    )


# Call the `initialize_chat_model` function to set up and assign a language model (LLM) to the variable `llm`
llm = initialize_chat_model()


def execute_sql_query(query: str):
    """
    Executes an SQL query on SQLserver and returns JSON-formatted results.
    If asked about the database schema, it will retrieve and return table structures.
    """
    try:
        # If the query is to show all tables in the database
        if query.strip().lower() == "show tables":
            tables = db.run("SELECT name FROM sys.tables;")
            return json.dumps({"tables": [table[0] for table in tables]}, indent=4)
        # If the query is to describe a specific table's schema
        elif query.strip().lower().startswith("describe "):
            table_name = query.split(" ", 1)[1].strip("")
            schema_info = db.run(
                f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}';")
            return json.dumps({"schema": schema_info}, indent=4)
        # Execute the user-provided SQL query
        result = db.run(query)

        # Identify if the query is an aggregation query (SUM, COUNT, AVG, MIN, MAX)
        # Define SQL keywords for aggregation
        aggregation_keywords = ["count(", "sum(", "avg(", "min(", "max("]
        is_aggregation_query = any(keyword in query.lower() for keyword in aggregation_keywords)

        # If result is a string, try evaluating to a Python object
        if isinstance(result, str):
            try:
                result = ast.literal_eval(result)  # Convert to a Python object if possible
            except Exception:
                pass  # If it fails, assume result is already JSON-compatible

        # Ensure that we correctly apply row limits
        if is_aggregation_query:
            # If it's an aggregation query, limit applies to number of results returned
            if isinstance(result, list) and len(result) > 100:
                return json.dumps({"message": "Try another query. Your request fetched more than 100 records."})

        else:
            # If the query is fetching rows (not just aggregation), check row count
            if isinstance(result, list) and len(result) > 100:
                return json.dumps({"message": "Try another query. Your request fetched more than 100 records."})
        if not result:
            return json.dumps({"message": "No data found."})
        # Return the data as JSON if within row limit
        return json.dumps({"data": result}, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)})


def create_sql_tool():
    """
    Creates and returns a LangChain tool for SQL execution.
    """
    return Tool(
        # Setting the name of tool to SQL Query Executor
        name="sql_Query_Executor",
        # Specify the function to execute SQL queries.
        func=execute_sql_query,
        # Description of tool
        description="Executes SQL Server queries and returns JSON"
                    "Retrieve schema details using 'SHOW TABLES;' (sys.tables in SQL Server) and INFORMATION_SCHEMA.COLUMNS"
                    "Ensure queries match the correct schema before execution"
                    "Skip rows with NULL values in calculations"
                    "For columns ending with 'Amt' or 'Amount', clean the values by removing commas and dollar signs using REPLACE(Column, ',', '') and REPLACE(Column, '$', ''). Convert only numeric values using CASE WHEN ISNUMERIC(Column) = 1 THEN CAST(REPLACE(REPLACE(Column, ',', ''), '$', '') AS FLOAT) ELSE NULL END"
                    "Use string functions (e.g., LEFT(column, N)) for filtering"
                    "If results exceed 100 rows or return 'Agent stopped due to iteration limit' or 'time limit', prompt user to refine the query"
                    "For queries returning multiple values:"
                    "If only one row with value, return it"
                    "If multiple rows share the value, return exactly the top 5 using ORDER BY column_name DESC with TOP 5"
                    "If more than 5 exist, return 5 and add, 'More records exist with the same highest value.'"
    )


# Call the `create_sql_tool` function to create a new LangChain tool for SQL execution and assign it to `SQL_tool`.
SQL_tool = create_sql_tool()


def fetch_full_schema():
    raw_tables = db.run("SELECT name FROM sys.tables;")

    try:
        tables = ast.literal_eval(raw_tables)  # Convert string to Python list
    except Exception as e:
        raise ValueError(f"Failed to parse table list: {e}")

    schema = {}
    formatted_schema=[]
    for table in tables:
        table_name = table[0]  # Safe because it's a tuple like ('ClaimsManagement',)
        columns_raw = db.run(
            f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'"
        )

        try:
            columns = ast.literal_eval(columns_raw) if isinstance(columns_raw, str) else columns_raw
        except Exception as e:
            columns = []
            print(f"Error parsing columns for {table_name}: {e}")

        schema[table_name] = columns
        formatted_columns = "\n  ".join([f"{col[0]} ({col[1]})" for col in columns])
        formatted_schema.append(f"Table: {table_name}\n  {formatted_columns}")
    return "\n\n".join(formatted_schema)


def create_agent():
    """
    Initializes and returns a LangChain agent for handling SQL queries.
    """
    return initialize_agent(
        # Pass a list containing the `SQL_tool` to the agent for executing SQL queries.
        tools=[SQL_tool],
        # Specify the language model (LLM) to be used by the agent
        llm=llm,
        # Use the OPENAI_FUNCTIONS agent type for handling interactions and tasks.
        agent=AgentType.OPENAI_FUNCTIONS,
        # Set verbosity to `False` to suppress detailed output logs.
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10,
        max_execution_time=120,
        system_message="You are a smart SQL assistant."
                       "Before querying data, check the database schema provided below.\n"
                        "If needed, use column names exactly as they appear. Use correct table relationships based on primary and foreign keys.\n"
                        "Strictly Avoid assumptions — only use the structure below.\n\n"
                       "### Database Schema ###\n"
                       f"Here is the schema you will work with:{fetch_full_schema()}" 
                       "When querying, use correct table relationships based on primary and foreign keys. "
                       "For example, if retrieving department projects, join 'projects' with 'departments' using 'dept_id'."
                       "Answer in one sentence"

                       "In case the prompt is very large and complex to be executed as SQLserver query, then return a message to the user to redefine the query in more structured way and narrow down the expected results"
                       "If the agent is returning this response:'It seems that I am unable to retrieve the schema details for[] table due to the limitations on the number of records returned', then return a message to the user to redefine the query for less output size."
                       "Do not return any hallucinated response, the reponse should be purely based on the provided database without any assumption"
    )


agent_executor = create_agent()


@app.route("/chat-sql-database", methods=["GET", "POST"])
def query_database():
    # Retrieve JSON data from the incoming request and store it in the variable `data`
    data = request.get_json()

    query = data.get("query", "").strip()  # Text input from the user

    # Validate query input
    if not query:
        return jsonify({"response": "Query cannot be empty. Please provide input!"}), 400

    # Handle termination commands in the backend
    if query.lower() in ["bye", "exit", "close"]:
        return jsonify({"response": "See you later!"}), 200

    # Process the query using agent_executor
    response = agent_executor.run(query)  # Process the query
    return jsonify({"response": response})  # Return the processed response as JSON


# For running the app of flask
if __name__ == "__main__":
    app.run(debug=True)

