from functions import check_vecdb, get_response
from fastapi import FastAPI, Form

app = FastAPI()


@app.post('/chatbot')
def financeaiwriter(title: str = Form(...)):
    ai_writer_template = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just 
    say "I don't know":

    <context>
    {context}
    </context>
    \n\n

    Now write a short post for the following topic:
    Question: "{Question}"
    Answer:"""

    persist_dir = "DB"
    documentsfolder="documents"
    db=check_vecdb(documentsfolder,persist_dir)
    docs = db.similarity_search(title)
    new_context = ""
    for document in docs:
        new_context += document.page_content

    response = get_response(title, new_context, ai_writer_template)
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)