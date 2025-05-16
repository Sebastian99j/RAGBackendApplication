from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from core.documents import documents
from services.retriever import get_context

prompt = PromptTemplate.from_template("Kontekst: {context}\nPytanie: {question}\nOdpowiedź:")
hf_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
hf_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
hf_pipe = pipeline("text-generation", model=hf_model, tokenizer=hf_tokenizer, max_new_tokens=100)
llm = HuggingFacePipeline(pipeline=hf_pipe)
chain = prompt | llm

def ask_langchain(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    return chain.invoke({"context": context, "question": question})

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.from_texts(documents, embedding_model)
retriever_faiss = faiss_index.as_retriever()

def ask_langchain_faiss(question: str):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_faiss)
    return qa_chain.run(question)
