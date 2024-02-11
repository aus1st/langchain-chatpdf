from dotenv import load_dotenv
import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter  import CharacterTextSplitter
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
def main():
    load_dotenv()
    #print(os.getenv("API_KEY"))
    st.set_page_config(page_title="Ask your PDF", page_icon=":book:", layout="wide")
    st.header("Ask your PDF 💬")
    
    #uploading the file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    
    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text +=  page.extract_text()
        
        #split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)
        
        #create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks,embeddings)

        #CHAT INPUT
        user_input = st.chat_input("Ask a question about your PDF")    
        if user_input is not None:
            docs = knowledge_base.similarity_search(user_input)
            #st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm,chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_input)
            
            st.chat_message(response)
if __name__ == "__main__":
    main()