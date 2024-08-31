from dotenv import load_dotenv
load_dotenv()
DATASET_PATH=os.getenv("DATASET_PATH")
import os
import textwrap
from dotenv import load_dotenv
# from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core import download_loader
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext
import re
import nest_asyncio
nest_asyncio.apply()
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings

from llama_index.embeddings.cohere import CohereEmbedding

# with input_typ='search_query'
cohere_api_key = os.getenv("COHERE_API_KEY")
embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",
)
# Settings.embed_model = HuggingFaceEmbedding(
#     model_name="BAAI/bge-small-en-v1.5"
# )
# Check for OpenAI API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("Groq API key not found in environment variables")
Settings.embed_model = embed_model
llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
Settings.llm = llm

def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)

def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)

def initialize_github_client():
    github_token = os.getenv("GITHUB_TOKEN")
    return GithubClient(github_token)



# Check for GitHub Token
github_token = os.getenv("GITHUB_TOKEN")
if not github_token:
    raise EnvironmentError("GitHub token not found in environment variables")

# Check for Activeloop Token
active_loop_token = os.getenv("ACTIVELOOP_TOKEN")
if not active_loop_token:
    raise EnvironmentError("Activeloop token not found in environment variables")


github_client = initialize_github_client()
download_loader("GithubRepositoryReader")

github_url = input("Please enter the GitHub repository URL: ")
owner, repo = parse_github_url(github_url)

while True:
    owner, repo = parse_github_url(github_url)
    if validate_owner_repo(owner, repo):
        loader = GithubRepositoryReader(
            github_client,
            owner=owner,
            repo=repo,
            filter_file_extensions=(
                [".py", ".js", ".ts", ".md", ".html", ".css", ".mjs", ".ejs"],
                GithubRepositoryReader.FilterType.INCLUDE,
            ),
            verbose=False,
            concurrent_requests=5,
        )
        print(f"Loading {repo} repository by {owner}")
        docs = loader.load_data(branch="main")
        print("Documents uploaded:")
        for doc in docs:
            print(doc.metadata)
        break  # Exit the loop once the valid URL is processed
    else:
        print("Invalid GitHub URL. Please try again.")
        github_url = input("Please enter the GitHub repository URL: ")

print("Uploading to vector store...")

# ====== Create vector store and upload data ======

vector_store = DeepLakeVectorStore(
    dataset_path=DATASET_PATH,
    overwrite=True,
    runtime={"tensor_db": True},
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
query_engine = index.as_query_engine()

# Include a simple question to test.
intro_question = "What is the repository about?"
print(f"Test question: {intro_question}")
print("=" * 50)
answer = query_engine.query(intro_question)

print(f"Answer: {textwrap.fill(str(answer), 100)} \n")
while True:
    user_question = input("Please enter your question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        print("Exiting, thanks for chatting!")
        break

    print(f"Your question: {user_question}")
    print("=" * 50)

    answer = query_engine.query(user_question)
    print(f"Answer: {textwrap.fill(str(answer), 100)} \n")