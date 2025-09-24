# -*- coding: utf-8 -*-
"""
문제 4.2: LangChain을 활용한 RAG 시스템 (수정된 코드)
"""

try:    
    from langchain_core.documents import Document as LangChainDocument    
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
    print("LangChain 라이브러리가 사용 가능합니다.")
except ImportError as e:
    print(e)
    LANGCHAIN_AVAILABLE = False
    print("LangChain 라이브러리가 설치되지 않았습니다.")


import json
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import re


@dataclass
class Document:
    """문서 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str

class MockEmbeddings:
    """임베딩 모의 클래스"""
    def __init__(self):
        self.dimension = 384
        self.word_vectors = {}

    def _text_to_vector(self, text: str) -> List[float]:
        text = text.lower().strip()
        words = re.findall(r'\w+', text)
        vector = np.zeros(self.dimension)
        if not words:
            return vector.tolist()

        for i, word in enumerate(words):
            if word in self.word_vectors:
                word_vector = self.word_vectors[word]
            else:
                np.random.seed(hash(word) % (2**31))
                word_vector = np.random.normal(0, 1, self.dimension)
                word_vector = word_vector / np.linalg.norm(word_vector)
                self.word_vectors[word] = word_vector

            weight = 1.0 / (i + 1)
            vector += weight * word_vector

        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._text_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._text_to_vector(text)

class MockVectorStore:
    """벡터 스토어 모의 클래스"""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []

    def add_documents(self, documents: List[Document]):
        texts = [doc.content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        self.documents.extend(documents)
        self.vectors.extend(vectors)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if not self.vectors:
            return []
        query_vector = np.array(self.embeddings.embed_query(query))
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            doc_vector_np = np.array(doc_vector)
            dot_product = np.dot(query_vector, doc_vector_np)
            norm_product = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector_np)
            similarity = dot_product / norm_product if norm_product > 0 else 0
            similarities.append((similarity, i))
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in similarities[:k]]
        return [self.documents[i] for i in top_indices]

class MockRAGChain:
    """RAG 체인 모의 클래스"""
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retrieval_k = 3

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs.get("query", "")
        relevant_docs = self.vector_store.similarity_search(query, k=self.retrieval_k)
        context = "\n\n".join([f"문서: {doc.content}" for doc in relevant_docs])
        answer = self._generate_answer(query, context, relevant_docs)
        return {"result": answer, "source_documents": relevant_docs}

    def _generate_answer(self, query: str, context: str, docs: List[Document]) -> str:
        query_lower = query.lower()
        if not docs:
            return "죄송하지만 관련 정보를 찾을 수 없습니다."
        
        first_doc_content = docs[0].content
        first_sentence = first_doc_content.split('.')[0]

        if "what" in query_lower or "무엇" in query_lower or "정의" in query_lower:
            return f"관련 정보에 따르면, {first_sentence.strip()}입니다."
        elif "how" in query_lower or "어떻게" in query_lower:
            return f"관련 문서('{docs[0].metadata.get('source')}')를 분석한 결과, '{first_sentence.strip()}'와 같이 설명할 수 있습니다."
        else:
            return f"검색된 {len(docs)}개의 관련 문서를 바탕으로 답변드립니다. 첫 번째 문서는 '{first_doc_content[:80]}...' 입니다."


class RAGSystem:
    """RAG 시스템 메인 클래스"""
    def __init__(self):
        self.sample_documents = [
            Document(content="인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 인식 등의 작업을 수행하는 기술입니다. 머신러닝과 딥러닝이 AI의 핵심 기술로 사용됩니다.", metadata={"source": "AI_intro.txt", "topic": "artificial_intelligence"}, doc_id="doc_001"),
            Document(content="머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 AI의 한 분야입니다. 지도학습, 비지도학습, 강화학습으로 나뉩니다.", metadata={"source": "ML_basics.txt", "topic": "machine_learning"}, doc_id="doc_002"),
            Document(content="딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 방법입니다. 이미지 인식, 자연어 처리, 음성 인식 등에 활용됩니다.", metadata={"source": "DL_guide.txt", "topic": "deep_learning"}, doc_id="doc_003"),
            Document(content="자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 AI 기술입니다. 토큰화, 구문 분석, 의미 분석 등의 과정을 통해 텍스트를 처리합니다.", metadata={"source": "NLP_overview.txt", "topic": "natural_language_processing"}, doc_id="doc_004"),
            Document(content="컴퓨터 비전은 디지털 이미지나 비디오에서 정보를 추출하고 분석하는 AI 분야입니다. 객체 탐지, 이미지 분류, 얼굴 인식 등이 주요 응용 분야입니다.", metadata={"source": "CV_intro.txt", "topic": "computer_vision"}, doc_id="doc_005"),
            Document(content="강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다. 게임 AI, 로봇 제어, 추천 시스템 등에 사용됩니다.", metadata={"source": "RL_basics.txt", "topic": "reinforcement_learning"}, doc_id="doc_006"),
            Document(content="트랜스포머는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처로, BERT, GPT 등 현대적인 언어 모델의 기반이 됩니다.", metadata={"source": "transformer_explained.txt", "topic": "transformer"}, doc_id="doc_007"),
            Document(content="BERT는 양방향 인코더 표현을 사용하는 트랜스포머 기반 언어 모델로, 문맥을 양방향으로 이해하여 높은 성능을 보입니다.", metadata={"source": "BERT_guide.txt", "topic": "bert"}, doc_id="doc_008")
        ]

    def setup_mock_rag(self):
        print("모의 RAG 시스템을 설정합니다...")
        embeddings = MockEmbeddings()
        vector_store = MockVectorStore(embeddings)
        vector_store.add_documents(self.sample_documents)
        rag_chain = MockRAGChain(vector_store)
        print("모의 RAG 시스템 설정 완료!")
        return rag_chain, vector_store

    def setup_real_rag(self):
        print("실제 RAG 시스템을 설정합니다...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # [수정] LangChain의 Document 형식으로 변환할 때 'LangChainDocument'와 'page_content' 사용
        langchain_docs = []
        for doc in self.sample_documents:
            langchain_doc = LangChainDocument(
                page_content=doc.content,
                metadata=doc.metadata
            )
            langchain_docs.append(langchain_doc)
            
        vector_store = FAISS.from_documents(langchain_docs, embeddings)
        rag_chain = RetrievalQA.from_chain_type(
            llm=None, # LLM이 없어도 검색은 가능
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True # 소스 문서 반환 옵션 추가
        )
        print("실제 RAG 시스템 설정 완료!")
        return rag_chain, vector_store

def create_rag_system():
    """RAG 시스템 생성"""
    system = RAGSystem()
    if LANGCHAIN_AVAILABLE:
        try:
            return system.setup_real_rag(), "Real RAG"
        except Exception as e:
            print(f"실제 RAG 시스템 생성 실패: {e}")
            print("대체 모의 RAG 시스템을 사용합니다.")
            return system.setup_mock_rag(), "Mock RAG"
    else:
        print("LangChain이 설치되지 않아 모의 RAG 시스템을 사용합니다.")
        return system.setup_mock_rag(), "Mock RAG"

def demonstrate_rag_queries():
    """RAG 시스템 질의 응답 데모"""
    print("\n=== RAG 시스템 질의 응답 데모 ===")
    print("문서 데이터베이스에서 관련 정보를 검색하여 답변합니다.")
    print("-" * 60)
    
    (rag_chain, vector_store), system_type = create_rag_system()
    print(f"사용 시스템: {system_type}")
    
    test_queries = [
        "인공지능이 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은?",
        "트랜스포머 아키텍처에 대해 설명해주세요",
        "자연어 처리는 어떤 과정을 거치나요?",
        "강화학습의 특징은?",
        "BERT 모델의 장점은?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- 질문 {i}: {query} ---")
        try:
            if system_type == "Mock RAG":
                result = rag_chain({"query": query})
                answer = result["result"]
                source_docs = result["source_documents"]
                print(f"답변: {answer}")
                print(f"\n참조 문서들:")
                for j, doc in enumerate(source_docs, 1):
                    # Mock 시스템은 자체 Document 클래스 사용 (content 속성)
                    print(f"  {j}. {doc.metadata.get('source', 'Unknown')}: {doc.content[:80]}...")
            
            else: # "Real RAG"
                # 실제 시스템은 LLM이 없으므로 직접 검색 후 결과 출력
                similar_docs = vector_store.similarity_search(query, k=3)
                print(f"답변: (LLM이 없어 검색 결과만 표시합니다)")
                print(f"\n검색된 관련 문서들:")
                for j, doc in enumerate(similar_docs, 1):
                    # [수정] LangChain Document의 텍스트 내용은 'page_content' 속성으로 접근
                    print(f"  {j}. {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:80]}...")
        
        except Exception as e:
            import traceback
            print(f"오류 발생: {e}")
            traceback.print_exc()

def main():
    print("=== 문제 4.2: LangChain을 활용한 RAG 시스템 ===")
    
    demonstrate_rag_queries()
    
    if not LANGCHAIN_AVAILABLE:
        print("\n=== 설치 안내 ===")
        print("실제 RAG 시스템을 사용하려면 아래 라이브러리를 설치하세요:")
        print("pip install langchain langchain-huggingface sentence-transformers faiss-cpu")
        print("pip install -U langchain-community")

if __name__ == "__main__":
    main()