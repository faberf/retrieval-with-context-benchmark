import argparse
import requests
import psycopg2

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Embed text data from PostgreSQL tables.")
parser.add_argument('--input_schema', type=str, required=True, help="Schema of the input table")
parser.add_argument('--input_table', type=str, required=True, help="Name of the input table")
parser.add_argument('--output_schema', type=str, required=True, help="Schema of the output table")
parser.add_argument('--output_table', type=str, required=True, help="Name of the output table")
args = parser.parse_args()

# Database connection settings
DB_HOST = "10.34.64.130"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "admin"

# Connect to PostgreSQL database
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

conn.autocommit = True
cur = conn.cursor()

def embed(text):
    url = "http://10.34.64.84:8888/api/legacy/tasks/text-embedding/e5mistral7b-instruct/jobs"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "text": text
    }
    
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses
    
    # Extract and return only the embedding list
    return response.json().get("embedding", [])

def fetch_rows(input_schema, input_table):
    # Define the query to fetch retrievableid and value columns
    query = f"""
    SELECT retrievableid, value
    FROM "{input_schema}".{input_table};
    """
    
    # Execute the query
    cur.execute(query)
    print("Query executed successfully.")  # Confirm query execution
    
    # Fetch all rows at once and iterate over them
    rows = cur.fetchall()
    if not rows:
        print("No rows found in the table.")
        
    for row in rows:
        yield row[0], row[1]

def ensure_vector_column_exists(output_schema, output_table):
    # Check if the 'vector' column exists
    query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = '{output_schema}'
    AND table_name = '{output_table}'
    AND column_name = 'vector';
    """
    cur.execute(query)
    
    # If the column does not exist, add it as a double precision array
    if not cur.fetchone():
        add_column_query = f"""
        ALTER TABLE "{output_schema}".{output_table}
        ADD COLUMN vector double precision[];
        """
        cur.execute(add_column_query)
        print("Added 'vector' column to the table.")

def update_rows(output_schema, output_table, retrievableid, embedding):
    # Convert the embedding list to a format suitable for PostgreSQL (e.g., an array)
    embedding_str = "{" + ",".join(map(str, embedding)) + "}"  # Embedding as a PostgreSQL array
    
    # Define the update query to set the vector column
    query = f"""
    UPDATE "{output_schema}".{output_table}
    SET vector = %s
    WHERE retrievableid = %s;
    """
    
    cur.execute(query, (embedding_str, retrievableid))

print("Ensuring 'vector' column exists...")
ensure_vector_column_exists(args.output_schema, args.output_table)

print("Fetching rows from the database...")
for retrievableid, value in fetch_rows(args.input_schema, args.input_table):
    print(f"Retrievable ID: {retrievableid}, Value: {value}")
    print("Embedding value...")
    embedding = embed(value)
    print(f"Embedding: {embedding}")
    print("Updating the vector column in the database...")
    update_rows(args.output_schema, args.output_table, retrievableid, embedding)
    print("Update complete.\n")  # Print a newline for separation

# Close the cursor and connection when done
cur.close()
conn.close()
