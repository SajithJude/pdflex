from flask import Flask, render_template, request, Response, stream_with_context, jsonify, send_from_directory
from flask_cors import CORS  # Import CORS
# from dotenv import load_dotenv

import webbrowser
from threading import Timer
import os
from llama_index.core import (
    # ServiceContext,
    # VectorStoreIndex,
    # SimpleDirectoryReader,
    load_index_from_storage,
    # set_global_service_context,
    StorageContext,
)
from llama_index.core import ServiceContext, set_global_service_context
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import QueryFusionRetriever




app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load environment variables
from creds import openai_key

openai_api_key = openai_key
os.environ["OPENAI_API_KEY"] = openai_key


if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")


# Set up the service context for llama-index with the desired OpenAI model
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4", temperature=0)
)
set_global_service_context(service_context)

# Load the data from the "data" directory
# data = SimpleDirectoryReader("data").load_data()
storage_context1 = StorageContext.from_defaults(persist_dir="data/storage")
# Create the index

index =load_index_from_storage(storage_context1)
QUERY_GEN_PROMPT = (
    "You are a helpful assistant tasked with generating search queries for comparative legal analysis based on section numbers. For each provided "
    "section number, generate 2 search queries that are aimed at retrieving information related to the section's name from both 'The Code of Criminal Procedure, 1973' "
    "and 'The Bhartiya Nagarik Suraksha Sanhita, 2023'. These queries should specifically seek to uncover details regarding the provisions or applications of these sections. "
    "The format for the search queries is as follows:\n"
    "Section Number: {section_number}\n"
    "Search Queries:\n"
    "1. What is the section name and content for section {section_number} in The Code of Criminal Procedure, 1973?\n"
    "2. What is the section name and content for section {section_number} in The Bhartiya Nagarik Suraksha Sanhita, 2023?\n"
)

retriever = QueryFusionRetriever(
    [index.as_retriever()],
    similarity_top_k=2,
    num_queries=2,  # set this to 1 to disable query generation
    use_async=True,
    verbose=True,
    # query_gen_prompt="",  # we could override the query generation prompt here
)
# Configure the chat engine with a memory buffer
memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "Act as an experienced risk and financial policy analyst"
        "You are now able to intelligently answer questions about the information you have been provided"
    ),
)


@app.route('/<filename>')
def serve_file(filename):
    # Ensure the filename ends with '.html' to serve only HTML files
    # if filename.endswith('.html'):
    return send_from_directory('templates', filename)
    # else:
    #     return "File not found", 404


@app.route('/')
def index():
    # page = request.args.get('page')  # Get the page parameter from the query string
    # if page is not None:
    #     return send_from_directory('templates', f'{page}.html')  # Render 5.html if page parameter is 5
    # else:
    return render_template('index.html')  # Render index.html for other cases

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    response = chat_engine.stream_chat(user_message)
    buffer = []
    buffer_size = 3

    def generate():
        for token in response.response_gen:
            buffer.append(token)
            if len(buffer) >= buffer_size:
                yield ''.join(buffer)
                buffer.clear()
        if buffer:
            yield ''.join(buffer)

    return Response(stream_with_context(generate()), content_type='text/plain')



@app.route('/retrive', methods=['POST'])
def retrive():
    user_message = request.json.get('message')
    nodes_with_scores = retriever.retrieve("41D")
    source = []
    for node in nodes_with_scores:
        source.append(f"Score: {node.score:.2f} - {node.text}...")

    return jsonify({"source":source})


    
    # response = chat_engine.stream_chat(user_message)
    # buffer = []
    # buffer_size = 3

    # def generate():
    #     for token in response.response_gen:
    #         buffer.append(token)
    #         if len(buffer) >= buffer_size:
    #             yield ''.join(buffer)
    #             buffer.clear()
    #     if buffer:
    #         yield ''.join(buffer)

    # return Response(stream_with_context(generate()), content_type='text/plain')


def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1, open_browser).start()
    app.run(debug=True)
