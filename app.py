from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

from flask_cors import CORS
from dotenv import load_dotenv


app = Flask(__name__)
CORS(app)
load_dotenv()
# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# PDF 업로드 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 벡터 저장소
vectorstore = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vectorstore
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # PDF 처리
            reader = PdfReader(filepath)
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

            # 텍스트 분할
            text_splitter = CharacterTextSplitter(
                separator = "\n",
                chunk_size = 1000,
                chunk_overlap  = 200,
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)

            # 임베딩 생성 및 저장
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(texts, embeddings)

            return jsonify({"message": "File uploaded and processed successfully"}), 200
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    global vectorstore
    if not vectorstore:
        return jsonify({"error": "No document uploaded yet"}), 400
    
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    docs = vectorstore.similarity_search(question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=question)

    return jsonify({"answer": response}), 200

if __name__ == '__main__':
    app.run(debug=True)