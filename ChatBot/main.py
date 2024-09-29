import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import faiss
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# Initialize global variables
vector_store = None
VECTOR_STORE_PATH = "vector_store"
pdf_list = []
is_loading = False

# Load the DistilBERT model for question answering
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
)

# Function to update status bar
def update_status(message):
    status_bar.config(text=message)

# Progress update helper
def update_progress(progress_value, max_value=100):
    progress['maximum'] = max_value
    progress['value'] = progress_value
    root.update_idletasks()

# Load PDF and create embeddings in the background
def load_pdf(pdf_path):
    global is_loading
    if is_loading:
        messagebox.showwarning("Warning", "PDF loading is already in progress.")
        return
    is_loading = True
    thread = threading.Thread(target=process_pdf_loading, args=(pdf_path,))
    thread.start()

# PDF loading process with progress updates
def process_pdf_loading(pdf_path):
    update_status("Loading PDF...")
    update_progress(10)  # Start progress
    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        update_status("Splitting document into chunks...")
        update_progress(30)  # Update progress
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(documents)

        update_status("Creating embeddings...")
        update_progress(60)
        embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

        global vector_store
        vector_store = FAISS.from_documents(split_documents, embeddings)

        update_status("Saving vector store...")
        save_vector_store(vector_store, embeddings)

        pdf_list.append(os.path.basename(pdf_path))
        pdf_listbox.insert(tk.END, os.path.basename(pdf_path))

        update_progress(100)  # Full progress
        update_status("PDF loaded successfully!")
        messagebox.showinfo("Success", "PDF loaded successfully!")
    except Exception as e:
        update_status("Error loading PDF.")
        messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
    finally:
        is_loading = False
        update_progress(0)

# Save FAISS vector store
def save_vector_store(vector_store, embeddings):
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)

    faiss.write_index(vector_store.index, os.path.join(VECTOR_STORE_PATH, "faiss_index"))

    with open(os.path.join(VECTOR_STORE_PATH, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    with open(os.path.join(VECTOR_STORE_PATH, "docstore.pkl"), "wb") as f:
        pickle.dump(vector_store.docstore, f)

    with open(os.path.join(VECTOR_STORE_PATH, "index_to_docstore_id.pkl"), "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

# Load FAISS vector store
def load_vector_store():
    global vector_store
    try:
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "faiss_index")):
            index = faiss.read_index(os.path.join(VECTOR_STORE_PATH, "faiss_index"))

            with open(os.path.join(VECTOR_STORE_PATH, "embeddings.pkl"), "rb") as f:
                embeddings = pickle.load(f)

            with open(os.path.join(VECTOR_STORE_PATH, "docstore.pkl"), "rb") as f:
                docstore = pickle.load(f)

            with open(os.path.join(VECTOR_STORE_PATH, "index_to_docstore_id.pkl"), "rb") as f:
                index_to_docstore_id = pickle.load(f)

            vector_store = FAISS(
                embeddings.embed_query,
                index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )

            update_status("Vector store loaded successfully!")
            messagebox.showinfo("Success", "Vector store loaded successfully!")
        else:
            update_status("No vector store found.")
    except Exception as e:
        update_status("Error loading vector store.")
        messagebox.showerror("Error", f"Failed to load vector store: {str(e)}")

# Upload PDF for processing
def upload_pdf():
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if pdf_path:
        load_pdf(pdf_path)

# Ask a question to the QA pipeline
def ask_question():
    user_query = question_entry.get("1.0", tk.END).strip()
    if not user_query:
        messagebox.showwarning("Input Error", "Please enter a question.")
        return

    if vector_store is None:
        messagebox.showerror("Error", "Please upload a PDF and load the vector store first.")
        return

    thread = threading.Thread(target=process_query, args=(user_query,))
    thread.start()

# Process the user's query
def process_query(user_query):
    update_status("Processing query...")
    try:
        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f"You: {user_query}\n")
        chat_display.config(state=tk.DISABLED)

        # Retrieve the most relevant document chunk(s)
        relevant_context = vector_store.similarity_search(user_query, k=5)  # Get more context

        # Concatenate the context to avoid excessive token usage
        context_text = "\n".join([doc.page_content[:500] for doc in relevant_context])

        # Using the QA pipeline to generate the answer
        result = qa_pipeline(question=user_query, context=context_text)

        # Extract the answer from the result
        answer = result['answer']

        # Validate response
        if not answer or len(answer.strip()) < 10:  # Check if response is meaningful
            answer = "I'm sorry, but I couldn't find a valid answer to your question. Can you please rephrase it?"

        # Friendly response formatting
        friendly_response = f"Here's what I found regarding your question: *{user_query}*\n\n{answer}"

        chat_display.config(state=tk.NORMAL)
        chat_display.insert(tk.END, f"Bot: {friendly_response}\n\n")
        chat_display.config(state=tk.DISABLED)
        update_status("Query processed successfully!")
    except Exception as e:
        update_status("Error processing query.")
        messagebox.showerror("Error", f"Failed to process query: {str(e)}")

# Clear the chat display
def clear_chat():
    chat_display.config(state=tk.NORMAL)
    chat_display.delete(1.0, tk.END)
    chat_display.config(state=tk.DISABLED)

# Initialize the Tkinter window
root = tk.Tk()
root.title("Enhanced RAG Chatbot")
root.geometry("600x600")

status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
progress.pack(pady=10)

load_vector_store()

upload_button = tk.Button(root, text="Upload PDF", command=upload_pdf)
upload_button.pack(pady=10)

pdf_listbox = tk.Listbox(root, height=5)
pdf_listbox.pack(pady=10)

question_entry = tk.Text(root, height=4, width=60)
question_entry.pack(pady=10)

ask_button = tk.Button(root, text="Ask Question", command=ask_question)
ask_button.pack(pady=5)

clear_button = tk.Button(root, text="Clear Chat", command=clear_chat)
clear_button.pack(pady=5)

chat_display = scrolledtext.ScrolledText(root, state='disabled', width=70, height=15)
chat_display.pack(pady=10)

root.mainloop()
