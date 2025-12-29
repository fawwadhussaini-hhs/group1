import os
import shutil
import subprocess
import gradio as gr

# Modern Chain Imports
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Core Utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

# from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Gemini & Google AI Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Get the directory of the current module
module_directory = os.path.dirname(os.path.abspath(__file__))


class DocumentProcessor:
    def __init__(self, document_paths, token):
        self.document_paths = document_paths
        


        os.environ["GOOGLE_API_KEY"] = "AIzaSyDlx3vUyWm0RObRtAqWizGbA28vxI4LkMo"
        # self.load_documents([r"E:\FreeLance\download\tmp_doc.pdf"])
        self.persist_directory = './docs/chroma/'
        # self.vectordb = None


        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview",
            temperature=1.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def load_documents(self, file_paths):
        self.document_paths = file_paths
        self.docs = []
        # FIX: Load ALL uploaded files, not just the first one
        for path in self.document_paths:
            loader = PyPDFLoader(path)
            self.docs.extend(loader.load())
        print('Document Loaded Successfully!')

    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        self.splits = text_splitter.split_documents(self.docs)

    def change_permissions(self, directory):
        try:
            # Define the command
            command = ["chmod", "777", "-R", directory]

            # Execute the command
            subprocess.run(command, check=True)

            print(f"Permissions for {directory} changed to 664 successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while changing permissions: {e}")

    def delete_embeddings(self):
        # FIX: Simple Windows-friendly directory removal
        if os.path.isdir(self.persist_directory):
            print('Cleaning up old database...')
            shutil.rmtree(self.persist_directory, ignore_errors=True)

    def create_embeddings(self):
        # embeddings = HuggingFaceEmbeddings()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectordb_doc = Chroma.from_documents(
            documents=self.splits,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        # print(self.vectordb_doc._collection.count())
        # self.vectordb = vectordb


    def get_embeddings(self):
        return self.vectordb_doc


    def parse_output(self, response):
        # Find the index where "Question:" starts
        question_index = response.find("Question:")
        # Get all text including and after "Question:"
        if question_index != -1:
            result_text = response[question_index:].strip()
            return result_text
        else:
            return "I apologies, I don't know the answer"

    def document_chain(self):
        prompt = ChatPromptTemplate.from_template("""
        ### System Instructions
        - You are a school administrator, you have to help students, parents and prospective customers.
        - You have to be polite and give answers based on provided document.
        - If you dont find the answer, route it to contact info@hhs.edu.pk
        ### Constraints
        - Use only the provided context to answer. 
        - If the answer is not contained within the context, strictly state: "I'm sorry, but the provided documents do not contain information to answer this question."
        - Do not use outside knowledge or make up facts.
        - Cite specific parts of the text if possible (e.g., "According to the document...").

        ### Context
        ---
        {context}
        ---

        ### User Question
        {input}

        ### Response:
        """)

        # Use the 'stuff' chain to pass all retrieved documents into this prompt
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        return document_chain


    def reterival_chain(self, document_chain, document_embeddings):
        retriever = document_embeddings.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain

    def get_response(self, retrieval_chain, message):
        # response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
        response = retrieval_chain.invoke({"input": message})
        # print(response["answer"])
        return response["answer"]


print('')


def upload_file(files, processor):
    try:
        file_paths = [file.name for file in files]

        processor.load_documents(file_paths)
        processor.split_documents()
        processor.delete_embeddings()
        doc_embeddings = processor.create_embeddings()
        gr.Info("Document Uploaded,Enjoy Chat Now!")
    except Exception as e:
        # Handle any exceptions that occur during execution
        print(f"An error occurred: {e}")
        gr.Warning("Upload File(s) Again!")
    # return doc_embeddings
    # print(file_paths)

def echo(message, history, processor):
    try:
        # Check if embeddings exist before proceeding
        document_embeddings = processor.get_embeddings()
        if not document_embeddings:
            return "Please upload a document first!"

        document_chain = processor.document_chain()
        retrieval_chain = processor.reterival_chain(document_chain, document_embeddings)

        response = retrieval_chain.invoke({"input": message})
        # FIX: Just return the answer; Gemini is usually concise
        return response["answer"]

    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request."

def upload_warning():
    gr.Warning("Upload PDF File(s) First!")


def main():
    css = """
    .container {
        height: 90vh;
    }

    .container_1 {
        height: 60vh;
    }

    .container_2 {
        height: 30vh;
    }
    """

    processor = DocumentProcessor(document_paths='', token='')
    with gr.Blocks(css=css) as demo:
        demo.load(upload_warning, inputs=None, outputs=None)
        with gr.Column(elem_classes=["container"]):
            gr.Markdown("## Chat with your Data")
            with gr.Column(elem_classes=["container_2"]):
                # gr.Markdown("Make sure uploading PDF file(s) first!")
                file_output = gr.File()
                upload_button = gr.UploadButton("Click to Upload File(s)", file_types=[".pdf", ".doc", ".docx"],
                                                file_count="multiple")

                # Function to handle the upload and pass the processor
                def process_upload(files):
                    upload_file(files, processor)

                # Get the document embeddings returned by process_upload
                upload_button.upload(process_upload, upload_button, file_output)

            with gr.Column(elem_classes=["container_1"]):
                def process_echo(message, history):
                    return echo(message, history, processor)

                custom_textbox = gr.Textbox(
                    placeholder="Type here...",
                    container=False,  # This is the most important "shrink" setting
                    scale=7  # Use scale instead of width/height
                )

                gr.ChatInterface(fn=process_echo,
                                 textbox=custom_textbox,
                                 examples=["Explain this concept", "what is summary of document", "create notes"])

                gr.Markdown("* Note: The answers can be incorrect, However they can be enhanced")

    demo.launch(theme = "ocean", share = True)


if __name__ == "__main__":

    main()
