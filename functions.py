from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#functions file
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
#to load environmental parameters
import os
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()
# Access the API key from the environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

# Check for database existence correctly
def check_vecdb(Documents_folder,persist_dir):
    if os.path.exists(persist_dir):  # Use os.path.exists to check directory existence
        # Load the existing ChromaDB
        database = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
        print("DB already exist")
    else:
        #if vector DB is not present then create divide files into chunks and create new embeddings
        document_directory = f"./{Documents_folder}"
        pdf_files = [os.path.join(document_directory, file) for file in os.listdir(document_directory) if file.endswith(".pdf")]

        # Loading all documents
        all_pages = []
        for pdf_file in pdf_files:
            global pages
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            all_pages.extend(pages)

        # Splitting documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False
        )
        data = text_splitter.split_documents(pages)

        # Creating Embeddings for generated Chunks
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai.api_key)
        # Storing Embedding and chunks into database
        database = Chroma.from_documents(data,
                                            embeddings,
                                            ids=[f"{item.metadata['source']} - {index}" for index, item in enumerate(data)],
                                            collection_name="docsEmbeddings",
                                            persist_directory=persist_dir
        )
        database.persist()
    return database
#This API will be used to generate response for AI writer.
def get_response(user_query, some_context, some_template):
    try:
        prompt = PromptTemplate(template=some_template, input_variables=['context', 'Question'])
        llm = ChatOpenAI(openai_api_key=openai.api_key, model="gpt-3.5-turbo-1106", temperature=0.8, max_tokens=128)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        
        with get_openai_callback() as cb:
            result = llm_chain.invoke({"context": some_context, "Question": user_query}, return_only_outputs=True)
        
        return result['text']
    except Exception as e:
        return f"Error occurred from LLM Response: {str(e)}"
