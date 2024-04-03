import os
import torch
from flask import Flask, request, jsonify, render_template
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import VectorStoreIndex, ServiceContext, load_index_from_storage, StorageContext
from llama_index.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

def initialize_llm():
    os.environ["CMAKE_ARGS"] = "LLAMA_CUBLAS=on"
    os.environ["FORCE_CMAKE"] = "1"
    
    llm = LlamaCPP(
        model_path="/home/manifold/Desktop/DAP/LIRNEgpt/llama_index/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        temperature=0.1,
        max_new_tokens=256,
        context_window=4096,
        model_kwargs={"n_gpu_layers": -1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=False,
    )
    return llm

def initialize_query_engine(llm, embed_model="local:BAAI/bge-small-en-v1.5", sentence_window_size=3, save_dir="./vector_store/index"):
    node_parser = SentenceWindowNodeParser(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text"
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir=save_dir),
        service_context=sentence_context,
    )
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=2, model="BAAI/bge-reranker-base"
    )
    engine = index.as_query_engine(
        similarity_top_k=6, node_postprocessors=[postproc, rerank]
    )
    return engine

app = Flask(__name__)

llm = initialize_llm()
query_engine = initialize_query_engine(llm)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['GET'])
def query():
    message = request.args.get('message', '')
    response = query_engine.query(message)
    response = str(response)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000, use_reloader=False)
    
