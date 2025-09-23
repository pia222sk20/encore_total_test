"""
문제 4.2: LangChain을 활용한 RAG 시스템

지시사항:
LangChain 라이브러리를 사용하여 Retrieval-Augmented Generation (RAG) 시스템을 
구축하세요. 문서 데이터베이스에서 관련 정보를 검색하고, 이를 바탕으로 
질문에 답변하는 시스템을 만들어보세요.
"""

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain.llms import HuggingFacePipeline
    from langchain.document_loaders import TextLoader
    from langchain.docstore.document import Document
    LANGCHAIN_AVAILABLE = True
    print("LangChain 라이브러리가 사용 가능합니다.")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain 라이브러리가 설치되지 않았습니다.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

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
        self.dimension = 384  # sentence-transformers 기본 차원
        self.word_vectors = {}  # 단어별 벡터 캐시
    
    def _text_to_vector(self, text: str) -> List[float]:
        """텍스트를 벡터로 변환 (단순 해시 기반)"""
        # 텍스트를 정규화
        text = text.lower().strip()
        words = re.findall(r'\\w+', text)
        
        # 단어들의 해시값을 기반으로 벡터 생성
        vector = np.zeros(self.dimension)
        
        for i, word in enumerate(words):
            if word in self.word_vectors:
                word_vector = self.word_vectors[word]
            else:
                # 단어 해시값으로 의사 랜덤 벡터 생성
                np.random.seed(hash(word) % (2**31))
                word_vector = np.random.normal(0, 1, self.dimension)
                word_vector = word_vector / np.linalg.norm(word_vector)  # 정규화
                self.word_vectors[word] = word_vector
            
            # 위치 가중치 적용
            weight = 1.0 / (i + 1)
            vector += weight * word_vector
        
        # 최종 정규화
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩"""
        return [self._text_to_vector(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리를 임베딩"""
        return self._text_to_vector(text)

class MockVectorStore:
    """벡터 스토어 모의 클래스"""
    
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        self.metadatas = []
    
    def add_documents(self, documents: List[Document]):
        """문서들을 벡터 스토어에 추가"""
        texts = [doc.content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)
        
        self.documents.extend(documents)
        self.vectors.extend(vectors)
        self.metadatas.extend([doc.metadata for doc in documents])
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """유사도 검색"""
        if not self.vectors:
            return []
        
        query_vector = np.array(self.embeddings.embed_query(query))
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            doc_vector = np.array(doc_vector)
            
            # 코사인 유사도
            dot_product = np.dot(query_vector, doc_vector)
            norm_product = np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
            else:
                similarity = 0
            
            similarities.append((similarity, i))
        
        # 유사도 순으로 정렬하고 상위 k개 선택
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_indices = [idx for _, idx in similarities[:k]]
        
        return [self.documents[i] for i in top_indices]

class MockRAGChain:
    """RAG 체인 모의 클래스"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.retrieval_k = 3
    
    def __call__(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """RAG 체인 실행"""
        query = inputs.get("query", "")
        
        # 1. 관련 문서 검색
        relevant_docs = self.vector_store.similarity_search(query, k=self.retrieval_k)
        
        # 2. 컨텍스트 구성
        context_pieces = []
        for doc in relevant_docs:
            context_pieces.append(f"문서: {doc.content}")
        
        context = "\\n\\n".join(context_pieces)
        
        # 3. 답변 생성 (규칙 기반 시뮬레이션)
        answer = self._generate_answer(query, context, relevant_docs)
        
        return {
            "result": answer,
            "source_documents": relevant_docs
        }
    
    def _generate_answer(self, query: str, context: str, docs: List[Document]) -> str:
        """답변 생성 시뮬레이션"""
        query_lower = query.lower()
        
        # 간단한 키워드 매칭 기반 답변 생성
        if "what" in query_lower or "무엇" in query_lower:
            if docs and docs[0].content:
                # 첫 번째 관련 문서에서 정보 추출
                content = docs[0].content
                sentences = content.split('.')
                if sentences:
                    return f"관련 정보에 따르면, {sentences[0].strip()}입니다."
        
        elif "how" in query_lower or "어떻게" in query_lower:
            return "관련 문서들을 분석한 결과, 단계별 방법이 설명되어 있습니다."
        
        elif "why" in query_lower or "왜" in query_lower:
            return "문서에서 찾은 정보를 바탕으로 그 이유를 설명드리겠습니다."
        
        elif "when" in query_lower or "언제" in query_lower:
            return "관련 문서에서 시기와 관련된 정보를 찾았습니다."
        
        else:
            if docs:
                return f"검색된 {len(docs)}개의 관련 문서를 바탕으로 답변드리겠습니다. {docs[0].content[:100]}..."
            else:
                return "죄송하지만 관련 정보를 찾을 수 없습니다."

class RAGSystem:
    """RAG 시스템 메인 클래스"""
    
    def __init__(self):
        self.sample_documents = [
            Document(
                content="인공지능(AI)은 컴퓨터 시스템이 인간의 지능을 모방하여 학습, 추론, 인식 등의 작업을 수행하는 기술입니다. 머신러닝과 딥러닝이 AI의 핵심 기술로 사용됩니다.",
                metadata={"source": "AI_intro.txt", "topic": "artificial_intelligence"},
                doc_id="doc_001"
            ),
            Document(
                content="머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 AI의 한 분야입니다. 지도학습, 비지도학습, 강화학습으로 나뉩니다.",
                metadata={"source": "ML_basics.txt", "topic": "machine_learning"},
                doc_id="doc_002"
            ),
            Document(
                content="딥러닝은 인공신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 방법입니다. 이미지 인식, 자연어 처리, 음성 인식 등에 활용됩니다.",
                metadata={"source": "DL_guide.txt", "topic": "deep_learning"},
                doc_id="doc_003"
            ),
            Document(
                content="자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리하는 AI 기술입니다. 토큰화, 구문 분석, 의미 분석 등의 과정을 통해 텍스트를 처리합니다.",
                metadata={"source": "NLP_overview.txt", "topic": "natural_language_processing"},
                doc_id="doc_004"
            ),
            Document(
                content="컴퓨터 비전은 디지털 이미지나 비디오에서 정보를 추출하고 분석하는 AI 분야입니다. 객체 탐지, 이미지 분류, 얼굴 인식 등이 주요 응용 분야입니다.",
                metadata={"source": "CV_intro.txt", "topic": "computer_vision"},
                doc_id="doc_005"
            ),
            Document(
                content="강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다. 게임 AI, 로봇 제어, 추천 시스템 등에 사용됩니다.",
                metadata={"source": "RL_basics.txt", "topic": "reinforcement_learning"},
                doc_id="doc_006"
            ),
            Document(
                content="트랜스포머는 어텐션 메커니즘을 기반으로 한 신경망 아키텍처로, BERT, GPT 등 현대적인 언어 모델의 기반이 됩니다.",
                metadata={"source": "transformer_explained.txt", "topic": "transformer"},
                doc_id="doc_007"
            ),
            Document(
                content="BERT는 양방향 인코더 표현을 사용하는 트랜스포머 기반 언어 모델로, 문맥을 양방향으로 이해하여 높은 성능을 보입니다.",
                metadata={"source": "BERT_guide.txt", "topic": "bert"},
                doc_id="doc_008"
            )
        ]
    
    def setup_mock_rag(self):
        """모의 RAG 시스템 설정"""
        print("모의 RAG 시스템을 설정합니다...")
        
        # 임베딩 모델 생성
        embeddings = MockEmbeddings()
        
        # 벡터 스토어 생성
        vector_store = MockVectorStore(embeddings)
        
        # 문서들을 벡터 스토어에 추가
        vector_store.add_documents(self.sample_documents)
        
        # RAG 체인 생성
        rag_chain = MockRAGChain(vector_store)
        
        print("RAG 시스템 설정 완료!")
        return rag_chain, vector_store
    
    def setup_real_rag(self):
        """실제 RAG 시스템 설정"""
        print("실제 RAG 시스템을 설정합니다...")
        
        try:
            # 임베딩 모델 로드
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 문서들을 LangChain Document 형식으로 변환
            langchain_docs = []
            for doc in self.sample_documents:
                langchain_doc = Document(
                    page_content=doc.content,
                    metadata=doc.metadata
                )
                langchain_docs.append(langchain_doc)
            
            # 벡터 스토어 생성
            vector_store = FAISS.from_documents(langchain_docs, embeddings)
            
            # RAG 체인 생성
            rag_chain = RetrievalQA.from_chain_type(
                llm=None,  # LLM이 없어도 검색은 가능
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            print("실제 RAG 시스템 설정 완료!")
            return rag_chain, vector_store
            
        except Exception as e:
            print(f"실제 RAG 설정 실패: {e}")
            return self.setup_mock_rag()

def create_rag_system():
    """RAG 시스템 생성"""
    
    system = RAGSystem()
    
    if LANGCHAIN_AVAILABLE:
        try:
            return system.setup_real_rag(), "Real RAG"
        except Exception as e:
            print(f"실제 RAG 시스템 생성 실패: {e}")
            print("모의 RAG 시스템을 사용합니다.")
            return system.setup_mock_rag(), "Mock RAG"
    else:
        print("LangChain이 설치되지 않아 모의 RAG 시스템을 사용합니다.")
        return system.setup_mock_rag(), "Mock RAG"

def demonstrate_rag_queries():
    """RAG 시스템 질의 응답 데모"""
    
    print("=== RAG 시스템 질의 응답 데모 ===")
    print("문서 데이터베이스에서 관련 정보를 검색하여 답변합니다.")
    print("-" * 60)
    
    (rag_chain, vector_store), system_type = create_rag_system()
    print(f"사용 시스템: {system_type}")
    
    # 테스트 질문들
    test_queries = [
        "인공지능이 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은?",
        "트랜스포머 아키텍처에 대해 설명해주세요",
        "자연어 처리는 어떤 과정을 거치나요?",
        "강화학습의 특징은?",
        "BERT 모델의 장점은?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n--- 질문 {i}: {query} ---")
        
        try:
            if system_type == "Mock RAG":
                # 모의 시스템 사용
                result = rag_chain({"query": query})
                answer = result["result"]
                source_docs = result["source_documents"]
                
                print(f"답변: {answer}")
                print(f"\\n참조 문서들:")
                for j, doc in enumerate(source_docs, 1):
                    print(f"  {j}. {doc.metadata.get('source', 'Unknown')}: {doc.content[:100]}...")
            
            else:
                # 실제 시스템 사용 (검색만)
                similar_docs = vector_store.similarity_search(query, k=3)
                print(f"검색된 관련 문서들:")
                for j, doc in enumerate(similar_docs, 1):
                    print(f"  {j}. {doc.metadata.get('source', 'Unknown')}: {doc.page_content[:100]}...")
        
        except Exception as e:
            print(f"오류 발생: {e}")

def analyze_retrieval_quality():
    """검색 품질 분석"""
    
    print("\\n=== 검색 품질 분석 ===")
    print("다양한 질문 유형에 대한 검색 성능 평가")
    print("-" * 60)
    
    (rag_chain, vector_store), system_type = create_rag_system()
    
    # 질문 유형별 테스트
    query_types = {
        "정의 질문": [
            "인공지능의 정의는?",
            "딥러닝이란 무엇인가요?"
        ],
        "비교 질문": [
            "머신러닝과 딥러닝의 차이점은?",
            "BERT와 트랜스포머의 관계는?"
        ],
        "방법 질문": [
            "자연어 처리는 어떻게 작동하나요?",
            "강화학습은 어떤 방식으로 학습하나요?"
        ],
        "응용 질문": [
            "컴퓨터 비전의 응용 분야는?",
            "트랜스포머가 사용되는 곳은?"
        ]
    }
    
    for query_type, queries in query_types.items():
        print(f"\\n--- {query_type} ---")
        
        for query in queries:
            print(f"\\n질문: {query}")
            
            if system_type == "Mock RAG":
                result = rag_chain({"query": query})
                source_docs = result["source_documents"]
                
                print(f"검색된 문서 수: {len(source_docs)}")
                if source_docs:
                    print(f"최상위 문서: {source_docs[0].metadata.get('topic', 'Unknown')}")
                    
                    # 관련성 평가 (키워드 기반)
                    query_keywords = set(query.lower().split())
                    doc_keywords = set(source_docs[0].content.lower().split())
                    overlap = len(query_keywords.intersection(doc_keywords))
                    print(f"키워드 겹침: {overlap}개")
            
            else:
                try:
                    similar_docs = vector_store.similarity_search(query, k=3)
                    print(f"검색된 문서 수: {len(similar_docs)}")
                    if similar_docs:
                        print(f"최상위 문서: {similar_docs[0].metadata.get('topic', 'Unknown')}")
                except Exception as e:
                    print(f"검색 오류: {e}")

def rag_system_components():
    """RAG 시스템 구성 요소 설명"""
    
    print("\\n=== RAG 시스템 구성 요소 ===")
    
    components = {
        "1. 문서 로더 (Document Loader)": {
            "역할": "다양한 형식의 문서를 시스템에 로드",
            "예시": "PDF, TXT, HTML, 웹페이지 등",
            "LangChain 클래스": "TextLoader, PyPDFLoader, WebBaseLoader"
        },
        "2. 텍스트 분할기 (Text Splitter)": {
            "역할": "긴 문서를 검색 가능한 청크로 분할",
            "예시": "문장 단위, 단락 단위, 토큰 수 기반",
            "LangChain 클래스": "RecursiveCharacterTextSplitter, TokenTextSplitter"
        },
        "3. 임베딩 모델 (Embedding Model)": {
            "역할": "텍스트를 벡터로 변환하여 의미적 유사도 계산",
            "예시": "sentence-transformers, OpenAI embeddings",
            "LangChain 클래스": "HuggingFaceEmbeddings, OpenAIEmbeddings"
        },
        "4. 벡터 스토어 (Vector Store)": {
            "역할": "임베딩 벡터를 저장하고 유사도 검색 수행",
            "예시": "FAISS, Chroma, Pinecone, Weaviate",
            "LangChain 클래스": "FAISS, Chroma, Pinecone"
        },
        "5. 검색기 (Retriever)": {
            "역할": "쿼리에 대해 관련 문서 청크 검색",
            "예시": "유사도 검색, 하이브리드 검색",
            "LangChain 클래스": "VectorStoreRetriever, MultiQueryRetriever"
        },
        "6. 언어 모델 (Language Model)": {
            "역할": "검색된 정보를 바탕으로 최종 답변 생성",
            "예시": "GPT, Claude, Llama",
            "LangChain 클래스": "OpenAI, HuggingFacePipeline"
        },
        "7. 체인 (Chain)": {
            "역할": "전체 RAG 파이프라인을 연결하고 조율",
            "예시": "RetrievalQA, ConversationalRetrievalChain",
            "LangChain 클래스": "RetrievalQA, RetrievalQAWithSourcesChain"
        }
    }
    
    for component, info in components.items():
        print(f"\\n{component}:")
        print(f"  역할: {info['역할']}")
        print(f"  예시: {info['예시']}")
        print(f"  LangChain: {info['LangChain 클래스']}")

def rag_optimization_techniques():
    """RAG 최적화 기법들"""
    
    print("\\n=== RAG 시스템 최적화 기법 ===")
    
    techniques = {
        "청크 크기 최적화": {
            "설명": "문서 분할 시 청크 크기를 작업에 맞게 조정",
            "방법": "100-500 토큰 범위에서 실험",
            "고려사항": "너무 작으면 컨텍스트 부족, 너무 크면 노이즈 증가"
        },
        "하이브리드 검색": {
            "설명": "키워드 검색과 벡터 검색을 결합",
            "방법": "BM25 + 벡터 유사도의 가중 평균",
            "고려사항": "정확한 매칭과 의미적 유사도 모두 활용"
        },
        "쿼리 확장": {
            "설명": "사용자 쿼리를 의미적으로 확장하여 검색 향상",
            "방법": "동의어, 관련 용어, 다른 표현 방식 추가",
            "고려사항": "과도한 확장은 노이즈 증가 가능"
        },
        "재순위화 (Re-ranking)": {
            "설명": "초기 검색 결과를 더 정교한 모델로 재순위화",
            "방법": "Cross-encoder 모델 사용",
            "고려사항": "계산 비용 증가하지만 정확도 향상"
        },
        "메타데이터 필터링": {
            "설명": "문서의 메타데이터를 활용한 검색 범위 제한",
            "방법": "날짜, 카테고리, 저자 등으로 필터링",
            "고려사항": "관련성 높은 문서에 집중 가능"
        },
        "문맥 압축": {
            "설명": "검색된 문서에서 관련 부분만 추출",
            "방법": "요약 모델이나 문장 선택 알고리즘 사용",
            "고려사항": "토큰 수 절약과 중요 정보 보존의 균형"
        }
    }
    
    for technique, info in techniques.items():
        print(f"\\n{technique}:")
        print(f"  설명: {info['설명']}")
        print(f"  방법: {info['방법']}")
        print(f"  고려사항: {info['고려사항']}")

def rag_evaluation_metrics():
    """RAG 평가 메트릭"""
    
    print("\\n=== RAG 시스템 평가 메트릭 ===")
    
    metrics = {
        "검색 품질": {
            "Recall@K": "상위 K개 결과에 정답이 포함된 비율",
            "Precision@K": "상위 K개 결과 중 관련 문서의 비율",
            "MRR": "첫 번째 관련 문서의 평균 역순위",
            "NDCG": "순위를 고려한 정규화된 할인 누적 이득"
        },
        "생성 품질": {
            "BLEU": "참조 답변과의 n-gram 유사도",
            "ROUGE": "요약 품질 측정 (recall 기반)",
            "BERTScore": "의미적 유사도 측정",
            "Faithfulness": "검색된 문서와의 일치도"
        },
        "종합 평가": {
            "End-to-End Accuracy": "전체 시스템의 정답률",
            "Human Evaluation": "인간 평가자의 품질 평가",
            "Response Time": "질의 응답 속도",
            "Cost Efficiency": "토큰 사용량 대비 성능"
        }
    }
    
    for category, metric_list in metrics.items():
        print(f"\\n{category}:")
        for metric, description in metric_list.items():
            print(f"  {metric}: {description}")

def practical_applications():
    """실제 RAG 활용 사례"""
    
    print("\\n=== RAG 시스템 실제 활용 사례 ===")
    
    applications = {
        "고객 지원": {
            "용도": "FAQ, 매뉴얼 기반 자동 응답",
            "데이터": "제품 문서, 과거 상담 기록",
            "효과": "24/7 지원, 일관된 답변 품질"
        },
        "법률 검색": {
            "용도": "판례, 법령 기반 법률 조언",
            "데이터": "법률 문서, 판례집",
            "효과": "빠른 법령 검색, 관련 판례 찾기"
        },
        "의료 진단 지원": {
            "용도": "의학 문헌 기반 진단 보조",
            "데이터": "의학 논문, 진료 가이드라인",
            "효과": "최신 연구 반영, 진단 정확도 향상"
        },
        "교육": {
            "용도": "교육 자료 기반 질의응답",
            "데이터": "교과서, 강의 노트, 참고서",
            "효과": "개인화된 학습 지원, 즉시 피드백"
        },
        "연구 지원": {
            "용도": "논문 데이터베이스 기반 연구 도움",
            "데이터": "학술 논문, 연구 보고서",
            "효과": "관련 연구 빠른 발견, 연구 동향 파악"
        },
        "기업 지식 관리": {
            "용도": "내부 문서 기반 정보 제공",
            "데이터": "내부 문서, 프로세스 매뉴얼",
            "효과": "지식 공유 효율화, 업무 생산성 향상"
        }
    }
    
    for domain, info in applications.items():
        print(f"\\n{domain}:")
        print(f"  용도: {info['용도']}")
        print(f"  데이터: {info['데이터']}")
        print(f"  효과: {info['효과']}")

def main():
    print("=== 문제 4.2: LangChain을 활용한 RAG 시스템 ===")
    
    # 1. RAG 질의 응답 데모
    demonstrate_rag_queries()
    
    # 2. 검색 품질 분석
    analyze_retrieval_quality()
    
    # 3. RAG 시스템 구성 요소
    rag_system_components()
    
    # 4. 최적화 기법들
    rag_optimization_techniques()
    
    # 5. 평가 메트릭
    rag_evaluation_metrics()
    
    # 6. 실제 활용 사례
    practical_applications()
    
    # 설치 안내
    if not LANGCHAIN_AVAILABLE:
        print("\\n=== 설치 안내 ===")
        print("실제 RAG 시스템을 사용하려면:")
        print("pip install langchain sentence-transformers faiss-cpu")

if __name__ == "__main__":
    main()