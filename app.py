from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter  import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def get_pdf_text(pdf):     
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text +=  page.extract_text()
    return text;    

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_stores(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    vector_store = FAISS.from_texts(text_chunks,embeddings)
    return vector_store

def get_conversation_chain(vector_store):
    llm = ""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    #print(os.getenv("API_KEY"))
    st.set_page_config(page_title="Ask your PDF", page_icon=":book:", layout="wide")
    st.header("Ask your PDF 💬")
    
    #uploading the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    with st.spinner('Processing...'):
    #extract the text
        if pdf is not None:
            text = get_pdf_text(pdf)
        
        #split text into chunks
            chunks = get_text_chunks(text)
        
        #create embeddings
            knowledge_base = get_vector_stores(chunks)

        #CHAT INPUT
            user_input = st.chat_input("Ask a question about your PDF")    
            if user_input is not None:
                vector_store = knowledge_base.similarity_search(user_input)
            #st.write(docs)

                #conversation chain
                st.session.conversation = get_conversation_chain(vector_store)
                llm = OpenAI()
                chain = load_qa_chain(llm,chain_type="stuff")
                response = chain.run(input_documents=vector_store, question=user_input)
            
                st.chat_message(response)
if __name__ == "__main__":
    main()