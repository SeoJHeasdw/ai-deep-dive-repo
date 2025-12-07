"""
RAG (Retrieval-Augmented Generation) 기초 파이프라인
Chapter 1: RAG 기초 - 청킹, 기본 파이프라인, 효과 비교

실습 항목:
1. 청킹(Chunking) 이해하기 - 텍스트 분할 전략
2. 기본 RAG 파이프라인 - 문서 -> 임베딩 -> 검색
3. RAG 있음 vs 없음 비교 - 답변 품질 차이

학습 목표:
- RAG가 무엇인지, 왜 필요한지 이해
- 청킹의 중요성과 전략 파악
- RAG 사용 시 답변 품질 향상 확인
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# 문서 파싱 라이브러리
import pdfplumber
from bs4 import BeautifulSoup
import markdown

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_subsection,
    print_key_points,
    get_openai_client,
    interpret_l2_distance,
    l2_distance_to_similarity
)

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


# ============================================================================
# 1. 문서 로더
# ============================================================================

class DocumentLoader:
    """다양한 형식의 문서를 로드"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """PDF 파일 로드 (한글 지원)"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """Markdown 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        # Markdown을 HTML로 변환 후 텍스트 추출
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    @staticmethod
    def load_html(file_path: str) -> str:
        """HTML 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """텍스트 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """파일 확장자에 따라 자동으로 로드"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return DocumentLoader.load_pdf(str(file_path))
        elif extension in ['.md', '.markdown']:
            return DocumentLoader.load_markdown(str(file_path))
        elif extension in ['.html', '.htm']:
            return DocumentLoader.load_html(str(file_path))
        elif extension == '.txt':
            return DocumentLoader.load_text(str(file_path))
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")


# ============================================================================
# 2. 텍스트 청킹
# ============================================================================

class TextChunker:
    """텍스트를 작은 청크로 분할"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크당 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _find_complete_list(self, text: str, start_pos: int, end_pos: int) -> tuple:
        """
        start_pos와 end_pos 사이에 리스트가 있으면, 
        그 리스트의 시작과 끝 위치를 반환
        
        Returns:
            (list_start, list_end) 또는 (None, None)
        """
        import re
        
        # 더 넓은 범위에서 리스트 찾기 (청크 크기의 2배까지)
        search_start = max(0, start_pos - 100)  # 이전 텍스트도 확인
        search_end = min(len(text), end_pos + self.chunk_size)  # 더 넓은 범위
        search_text = text[search_start:search_end]
        
        # 번호 매겨진 리스트 패턴 개선
        # 1. 로 시작하는 리스트를 찾고, 연속된 번호들을 모두 포함
        pattern = r'(?:^|\n)(1\.\s.+?)(?=\n(?!\d+\.\s)|\Z)'
        
        # 첫 번째 리스트 항목 찾기
        first_match = re.search(r'(?:^|\n)(1\.\s)', search_text, re.MULTILINE)
        if first_match:
            list_start_in_search = first_match.start()
            if first_match.group().startswith('\n'):
                list_start_in_search += 1
            
            # 연속된 번호 찾기
            current_pos = list_start_in_search
            last_number = 1
            list_end_in_search = current_pos
            
            while current_pos < len(search_text):
                # 다음 번호 찾기
                next_pattern = rf'(?:^|\n)({last_number + 1}\.\s)'
                next_match = re.search(next_pattern, search_text[current_pos:], re.MULTILINE)
                
                if next_match and next_match.start() < 200:  # 항목 간 거리 제한
                    # 다음 번호의 끝 찾기
                    next_start = current_pos + next_match.start()
                    if search_text[next_start] == '\n':
                        next_start += 1
                    
                    # 해당 항목의 끝 찾기 (다음 번호 또는 빈 줄까지)
                    item_end = search_text.find('\n\n', next_start)
                    next_item = re.search(rf'(?:^|\n){last_number + 2}\.\s', 
                                         search_text[next_start:], re.MULTILINE)
                    
                    if next_item and (item_end == -1 or next_start + next_item.start() < item_end):
                        item_end = next_start + next_item.start()
                    elif item_end == -1:
                        item_end = len(search_text)
                    
                    list_end_in_search = item_end
                    current_pos = next_start
                    last_number += 1
                else:
                    # 마지막 항목의 끝 찾기
                    item_end = search_text.find('\n\n', current_pos)
                    if item_end == -1:
                        item_end = search_text.find('\n', current_pos + 10)
                        if item_end == -1:
                            item_end = len(search_text)
                    list_end_in_search = item_end
                    break
            
            list_start = search_start + list_start_in_search
            list_end = search_start + list_end_in_search
            
            # 리스트가 청크 범위와 겹치면 반환
            if list_start < end_pos and list_end > start_pos:
                return (list_start, list_end)
        
        # 불릿 리스트 패턴
        pattern = r'((?:^[-*•]\s.+$\n?)+)'
        matches = list(re.finditer(pattern, search_text, re.MULTILINE))
        
        if matches:
            # 청크 범위와 가장 많이 겹치는 리스트 찾기
            best_match = None
            best_overlap = 0
            
            for match in matches:
                match_start = search_start + match.start()
                match_end = search_start + match.end()
                
                # 청크 범위와의 겹침 계산
                overlap_start = max(start_pos, match_start)
                overlap_end = min(end_pos, match_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (match_start, match_end)
            
            if best_match:
                return best_match
        
        return (None, None)
    
    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할 (리스트 인식 개선)"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # 텍스트 끝이면 그냥 추가
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 문장 경계 찾기 (우선순위 순서)
            best_end = -1
            
            # 0순위: 리스트 확인 (현재 청크 내에 리스트가 있으면 완전히 포함)
            list_start, list_end = self._find_complete_list(text, start, end)
            if list_start is not None and list_end is not None:
                # 리스트가 현재 청크 범위에 걸쳐있는 경우
                if list_start < end:
                    # 리스트가 청크 초반에 시작하는 경우
                    if list_start < start + self.chunk_size * 0.4:  # 청크 40% 이내
                        # 리스트 끝까지 포함
                        best_end = list_end
                    # 리스트가 청크 후반에 시작하는 경우  
                    else:
                        # 리스트 시작 전의 적절한 위치에서 자르기
                        # 리스트 헤더를 찾기
                        header_pos = text.rfind('\n', max(0, list_start - 50), list_start)
                        if header_pos != -1 and header_pos > start + self.chunk_size * 0.3:
                            best_end = header_pos + 1
                        else:
                            # 단락 끝 찾기
                            para_end = text.rfind('\n\n', start, list_start)
                            if para_end != -1 and para_end > start + self.chunk_size * 0.3:
                                best_end = para_end + 2
            
            # 1순위: 단락 끝 (빈 줄)
            if best_end == -1:
                double_newline = text.rfind('\n\n', start, end + 50)
                if double_newline != -1:
                    best_end = double_newline + 2
            
            # 2순위: 완전한 문장 끝 (마침표 + 줄바꿈 + 대문자/숫자로 시작)
            if best_end == -1:
                import re
                for i in range(end, max(start, end - 100), -1):
                    if i < text_length - 1 and text[i] == '.' and text[i+1] == '\n':
                        next_line_start = i + 2
                        if next_line_start < text_length:
                            # 다음 줄 확인
                            line_start = next_line_start
                            line_text = text[line_start:min(line_start + 20, text_length)]
                            
                            # 리스트 항목이 아닌지 확인
                            is_list = re.match(r'^\d+\.\s|^[-*•]\s', line_text)
                            
                            if not is_list:
                                next_char = text[next_line_start]
                                if next_char.isupper() or next_char.isdigit() or next_char == '\n':
                                    best_end = i + 2
                                    break
            
            # 3순위: 마침표 + 공백
            if best_end == -1:
                period_space = text.rfind('. ', start, end + 30)
                if period_space != -1:
                    best_end = period_space + 2
            
            # 4순위: 느낌표/물음표
            if best_end == -1:
                for punct in ['! ', '? ', '。']:
                    pos = text.rfind(punct, start, end + 30)
                    if pos != -1:
                        best_end = pos + len(punct)
                        break
            
            # 5순위: 줄바꿈
            if best_end == -1:
                newline = text.rfind('\n', start, end + 20)
                if newline != -1:
                    best_end = newline + 1
            
            # 6순위: 공백
            if best_end == -1:
                space = text.rfind(' ', start, end)
                if space != -1 and space > start + self.chunk_size // 2:
                    best_end = space + 1
            
            # 최종: 어쩔 수 없이 강제로 자르기
            if best_end == -1:
                best_end = end
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 다음 청크 시작 위치 결정
            # 리스트가 완전히 포함된 경우, 오버랩 없이 시작
            if list_start is not None and list_end is not None and best_end == list_end:
                # 리스트 다음부터 시작 (오버랩 없음)
                next_start = best_end
            else:
                # 일반적인 경우 오버랩 적용
                next_start = best_end - self.chunk_overlap
            
            # 진행이 없으면 강제로 앞으로 (무한 루프 방지)
            if next_start <= start:
                next_start = best_end
            
            start = next_start
        
        return chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[str]:
        """문장 단위로 청크 분할"""
        # 간단한 문장 분리 (더 정교한 방법은 nltk.sent_tokenize 사용)
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks


# ============================================================================
# 3. RAG 시스템
# ============================================================================

class RAGSystem:
    """RAG 시스템 전체 파이프라인"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """RAG 시스템 초기화"""
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client()
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
        
        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        
        # 문서 로더 및 청커
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성""" 
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def add_text(self, text: str, source_name: str = "sample_text", 
                 metadata: Optional[Dict] = None, check_duplicate: bool = True) -> int:
        """
        텍스트를 직접 청킹하여 Vector DB에 저장 (PDF 없이 실습 가능)
        
        Args:
            text: 추가할 텍스트
            source_name: 소스 이름
            metadata: 추가 메타데이터
            check_duplicate: 중복 체크 여부
        
        Returns:
            추가된 청크 수
        """
        print(f"\n[DOC] 텍스트 로딩: {source_name}")
        
        # 중복 체크
        if check_duplicate:
            try:
                existing = self.collection.get(
                    where={"source": source_name}
                )
                if existing and existing['ids']:
                    print(f"   이미 추가된 문서입니다: {source_name}")
                    print(f"   기존 청크 수: {len(existing['ids'])}개")
                    return 0
            except Exception as e:
                # 중복 체크 실패는 무시하고 계속 진행
                pass
        
        print(f"   텍스트 크기: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunker.chunk_text(text)
        print(f"   청크 수: {len(chunks)}개")
        
        # 청크 미리보기
        print(f"\n{'─'*60}")
        print("[CHUNK] 청크 미리보기:")
        print(f"{'─'*60}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[청크 {i+1}] (길이: {len(chunk)} 문자)")
            import re
            has_list = bool(re.search(r'^\d+\.\s|^[-*•]\s', chunk, re.MULTILINE))
            if has_list:
                print("   [v] 리스트 포함")
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            for line in preview.split('\n')[:5]:
                print(f"   {line}")
        
        if len(chunks) > 3:
            print(f"\n   ... (나머지 {len(chunks) - 3}개 청크 생략)")
        print(f"{'─'*60}")
        
        # 임베딩 생성
        print("\n[...] 임베딩 생성 중...")
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Vector DB에 저장
        start_idx = self.collection.count()
        ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        metadata['source'] = source_name
        
        metadatas = [metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
            meta['chunk_index'] = i
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"[OK] {len(chunks)}개 청크 저장 완료")
        return len(chunks)
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None, 
                     check_duplicate: bool = True, use_sample_if_short: bool = True) -> int:
        """
        문서를 로드하고 청킹하여 Vector DB에 저장
        PDF 내용이 짧으면 자동으로 샘플 텍스트 사용
        
        Args:
            file_path: 문서 파일 경로
            metadata: 추가 메타데이터
            check_duplicate: 중복 체크 여부
            use_sample_if_short: 내용이 짧으면 샘플 텍스트 사용
        
        Returns:
            추가된 청크 수
        """
        print(f"\n[FILE] 문서 로딩: {file_path}")
        
        # 1. 문서 로드
        document_text = self.loader.load_document(file_path)
        
        # 2. 텍스트 길이 확인 및 샘플 텍스트로 대체
        if use_sample_if_short:
            text, source_type = get_sample_or_document_text(document_text)
            if source_type == "sample":
                return self.add_text(text, source_name="AI_가이드", 
                                    metadata=metadata, check_duplicate=check_duplicate)
        else:
            text = document_text
        
        # 중복 체크
        if check_duplicate:
            try:
                existing = self.collection.get(
                    where={"source": str(file_path)}
                )
                if existing and existing['ids']:
                    print(f"   이미 추가된 문서입니다: {file_path}")
                    print(f"   기존 청크 수: {len(existing['ids'])}개")
                    return 0
            except Exception as e:
                # 중복 체크 실패는 무시하고 계속 진행
                pass
        
        print(f"   문서 크기: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunker.chunk_text(text)
        print(f"   청크 수: {len(chunks)}개")
        
        # 청크 미리보기
        print(f"\n{'─'*60}")
        print("[CHUNK] 청크 미리보기:")
        print(f"{'─'*60}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[청크 {i+1}] (길이: {len(chunk)} 문자)")
            import re
            has_list = bool(re.search(r'^\d+\.\s|^[-*•]\s', chunk, re.MULTILINE))
            if has_list:
                print("   [v] 리스트 포함")
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            for line in preview.split('\n')[:5]:
                print(f"   {line}")
        
        if len(chunks) > 3:
            print(f"\n   ... (나머지 {len(chunks) - 3}개 청크 생략)")
        print(f"{'─'*60}")
        
        # 임베딩 생성
        print("\n[...] 임베딩 생성 중...")
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Vector DB에 저장
        start_idx = self.collection.count()
        ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        metadata['source'] = str(file_path)
        
        metadatas = [metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
            meta['chunk_index'] = i
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"[OK] {len(chunks)}개 청크 저장 완료")
        return len(chunks)
    
    def search(self, query: str, n_results: int = 3) -> List[SearchResult]:
        """
        쿼리와 유사한 문서 검색
        
        Returns:
            검색 결과 리스트 (거리 기반 점수로 정렬)
            
        Note:
            - ChromaDB는 L2 거리를 반환합니다
            - 점수 = 1/(1+거리)로 0~1 범위로 정규화
            - ⚠️ 이것은 코사인 유사도가 아닙니다!
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        search_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # L2 거리를 0~1 점수로 변환
            # 낮은 거리 = 높은 점수 (가까울수록 관련성 높음)
            # ⚠️ 주의: 이것은 코사인 유사도가 아님!
            normalized_score = 1 / (1 + dist)
            search_results.append(SearchResult(
                content=doc,
                score=normalized_score,
                metadata={**meta, "raw_distance": dist},
                rank=i + 1
            ))
        
        return search_results
    
    def generate_answer_without_rag(self, question: str) -> str:
        """RAG 없이 LLM에게 직접 질문"""
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def generate_answer_with_rag(self, question: str, n_results: int = 3) -> Dict:
        """RAG를 사용하여 답변 생성"""
        # 1. 관련 문서 검색
        search_results = self.search(question, n_results=n_results)
        
        # 2. 컨텍스트 구성
        context = "\n\n".join([r.content for r in search_results])
        
        # 3. 프롬프트 구성
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {question}

답변:"""
        
        # 4. LLM 응답 생성
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            'answer': response.choices[0].message.content,
            'context': context,
            'sources': [r.metadata for r in search_results],
            'distances': [r.metadata.get('raw_distance', 0) for r in search_results],
            'search_results': search_results
        }
    
    def evaluate_retrieval(self, query: str, relevant_chunk_ids: List[int], 
                          k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        검색 품질 평가 (Recall@K)
        
        Args:
            query: 검색 쿼리
            relevant_chunk_ids: 정답으로 간주되는 청크 인덱스 리스트
            k_values: 평가할 K 값들
        
        Returns:
            각 K값에 대한 Recall 점수
        
        Note:
            Recall@K = (K개 검색 결과 중 정답 개수) / (전체 정답 개수)
            - 1.0: 모든 정답을 찾음
            - 0.0: 정답을 하나도 못 찾음
        """
        max_k = max(k_values)
        search_results = self.search(query, n_results=max_k)
        
        # 검색된 청크의 인덱스 추출
        retrieved_ids = []
        for result in search_results:
            chunk_idx = result.metadata.get('chunk_index')
            if chunk_idx is not None:
                retrieved_ids.append(chunk_idx)
        
        # 각 K값에 대한 Recall 계산
        recalls = {}
        for k in k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_chunk_ids)
            
            if len(relevant_set) == 0:
                recalls[f'recall@{k}'] = 0.0
            else:
                hits = len(retrieved_at_k & relevant_set)
                recalls[f'recall@{k}'] = hits / len(relevant_set)
        
        return recalls
    
    def check_hallucination(self, answer: str, context: str, 
                           check_terms: List[str] = None) -> Dict[str, Any]:
        """
        환각(Hallucination) 감지
        
        Args:
            answer: LLM이 생성한 답변
            context: 제공된 컨텍스트
            check_terms: 확인할 특정 용어들 (예: ["RAG", "Retrieval"])
        
        Returns:
            환각 감지 결과
        
        Note:
            간단한 규칙 기반 감지입니다. 실무에서는 더 정교한 방법 필요:
            - LLM-as-Judge 사용
            - NLI (Natural Language Inference) 모델 사용
            - 팩트 체킹 API 활용
        """
        result = {
            'potential_hallucinations': [],
            'verified_terms': [],
            'hallucination_risk': 'low'
        }
        
        # 기본 검사 용어 (RAG 관련 흔한 오류)
        if check_terms is None:
            check_terms = [
                ("Recall-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
                ("Recollection-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
                ("Retrieve-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
            ]
        
        # 잘못된 용어 검사
        for wrong, correct, desc in check_terms:
            if wrong.lower() in answer.lower():
                result['potential_hallucinations'].append({
                    'found': wrong,
                    'expected': correct,
                    'description': desc
                })
        
        # 컨텍스트에 없는 고유명사/숫자 검사 (간단한 휴리스틱)
        import re
        
        # 답변에서 연도 추출
        years_in_answer = set(re.findall(r'\b(19|20)\d{2}\b', answer))
        years_in_context = set(re.findall(r'\b(19|20)\d{2}\b', context))
        
        unverified_years = years_in_answer - years_in_context
        if unverified_years:
            result['potential_hallucinations'].append({
                'found': list(unverified_years),
                'expected': '컨텍스트에 있는 연도',
                'description': '컨텍스트에 없는 연도 언급'
            })
        
        # 리스크 레벨 결정
        if len(result['potential_hallucinations']) >= 2:
            result['hallucination_risk'] = 'high'
        elif len(result['potential_hallucinations']) == 1:
            result['hallucination_risk'] = 'medium'
        
        return result


# ============================================================================
# 출력 포맷팅 유틸리티
# ============================================================================

def format_chunk(content: str, indent: str = "      ") -> str:
    """청크 내용을 보기 좋게 포맷팅"""
    lines = content.strip().split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(f"{indent}{line}")
    return '\n'.join(formatted_lines)


def format_chunk_id(metadata: Dict) -> str:
    """청크 ID를 일관되게 포맷팅 (1-based indexing)"""
    chunk_idx = metadata.get('chunk_index', '?')
    if isinstance(chunk_idx, int):
        return str(chunk_idx + 1)  # 0-based → 1-based
    return str(chunk_idx)


def print_search_result(result: SearchResult, index: int, show_full: bool = True):
    """검색 결과를 포맷팅하여 출력"""
    chunk_id = format_chunk_id(result.metadata)
    
    print(f"  [{index}] 점수: {result.score:.4f} | 청크 #{chunk_id} ({len(result.content)}자)")
    
    if show_full:
        print(f"  {'─'*50}")
        print(format_chunk(result.content))
        print(f"  {'─'*50}")
    else:
        preview = result.content.replace('\n', ' ')[:100]
        print(f"      {preview}...")


def find_sample_document():
    """샘플 문서 찾기 (PDF 우선, 없으면 샘플 텍스트 사용)"""
    current_dir = Path(__file__).parent
    
    # PDF 파일 찾기
    pdf_files = list(current_dir.glob("*.pdf"))
    if pdf_files:
        print(f"[FILE] PDF 파일 발견: {pdf_files[0].name}")
        return pdf_files[0], "pdf"
    
    # PDF가 없으면 샘플 텍스트 사용
    print("[DOC] PDF 파일이 없습니다. 내장 샘플 텍스트를 사용합니다.")
    return None, "sample"


# ============================================================================
# 데모 함수들
# ============================================================================

def demo_chunking():
    """실습 1: 청킹(Chunking) 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: 청킹(Chunking) 이해하기")
    print("="*80)
    print("목표: 텍스트를 작은 청크로 분할하는 전략 이해")
    print("핵심: 청크 크기와 오버랩이 검색 품질에 미치는 영향")
    
    # 샘플 텍스트 (간단한 예시)
    sample_text = """인공지능(AI)은 인간의 학습, 추론, 지각 능력을 컴퓨터로 구현하는 기술입니다. 1956년 다트머스 회의에서 존 매카시가 처음으로 "인공지능"이라는 용어를 사용했습니다. 인공지능은 약한 AI, 강한 AI, 초인공지능으로 분류됩니다. 약한 AI는 특정 작업에 특화된 AI이고, 강한 AI는 인간 수준의 범용 지능을 목표로 합니다. 머신러닝은 AI의 하위 분야로, 데이터에서 패턴을 학습하여 예측합니다. 딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 기술입니다."""
    
    print(f"\n원본 텍스트 (길이: {len(sample_text)} 문자):")
    print(f"{'─'*60}")
    print(sample_text)
    print(f"{'─'*60}")
    
    # 다양한 청크 크기로 테스트
    chunk_configs = [
        (100, 20, "작은 청크"),
        (200, 30, "중간 청크"),
        (300, 40, "큰 청크"),
    ]
    
    for chunk_size, overlap, desc in chunk_configs:
        print_section_header(f"청크 크기: {chunk_size}자, 오버랩: {overlap}자 ({desc})")
        
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunker.chunk_text(sample_text)
        
        print(f"생성된 청크 수: {len(chunks)}개")
        
        # 오버랩으로 인한 중복 설명
        total_chars = sum(len(c) for c in chunks)
        if total_chars > len(sample_text):
            overlap_chars = total_chars - len(sample_text)
            print(f"[참고] 오버랩으로 인한 중복: 약 {overlap_chars}자")
            print(f"       (오버랩 {overlap}자 × 경계 {len(chunks)-1}개 ≈ {overlap*(len(chunks)-1)}자)")
        print()
        
        for i, chunk in enumerate(chunks):
            print(f"[청크 {i+1}] (길이: {len(chunk)} 문자)")
            print(f"'{chunk}'")
            print()
    
    # 요약
    print("="*60)
    print("[TIP] 청킹 핵심 포인트:")
    print("="*60)
    print("  - 청크 크기가 작을수록 -> 더 많은 청크, 정확한 검색, API 비용 증가")
    print("  - 청크 크기가 클수록 -> 더 적은 청크, 풍부한 문맥, 노이즈 가능")
    print("  - 오버랩 -> 청크 간 문맥 연결 유지, 정보 손실 방지")
    print("  - 권장: 도메인에 따라 300~1000자 실험 후 결정")
    
    print("""
  ────────────────────────────────────────────────────────────
  [실무 참고] 다양한 청킹 기법
  ────────────────────────────────────────────────────────────
  이 실습에서는 "문자 기반 청킹"을 사용했습니다.
  실무에서는 다음 기법들도 자주 사용됩니다:
  
  1. 문장 단위 청킹 (Sentence Split)
     * 문장 경계에서 분할 → 의미 단절 최소화
     * NLTK sent_tokenize, SpaCy 등 활용
  
  2. 의미 단위 청킹 (Semantic Chunking)
     * 임베딩 유사도로 "의미가 바뀌는 지점" 감지
     * 같은 주제끼리 묶음 → 검색 품질 향상
     * LangChain SemanticChunker 등 활용
  
  3. 문서 구조 기반 청킹
     * 마크다운 헤더, HTML 태그 등 구조 활용
     * 섹션/챕터 단위로 자연스럽게 분할
  
  [TIP] 선택 기준:
  * 빠른 구현 → 문자 기반 (이 실습)
  * 품질 우선 → 의미 단위 청킹
  * 구조화된 문서 → 문서 구조 기반
  ────────────────────────────────────────────────────────────
    """)


def demo_basic_rag():
    """실습 2: 기본 RAG 파이프라인"""
    print("\n" + "="*80)
    print("[2] 실습 2: 기본 RAG 파이프라인")
    print("="*80)
    print("목표: 문서 -> 임베딩 -> 검색 -> 답변 생성 과정 이해")
    print("핵심: 벡터 검색으로 관련 문서를 찾아 LLM에게 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 샘플 문서 찾기
    sample_file, source_type = find_sample_document()
    
    # RAG 시스템 초기화 (새 컬렉션)
    rag = RAGSystem("demo_rag_basic")
    
    # 문서 추가 (PDF 또는 샘플 텍스트)
    if source_type == "pdf":
        rag.add_document(str(sample_file), 
                        metadata={"type": "tutorial", "topic": "AI"}, 
                        check_duplicate=True)
    else:
        rag.add_text(SAMPLE_TEXT, source_name="AI_가이드",
                    metadata={"type": "tutorial", "topic": "AI"},
                    check_duplicate=True)
    
    # 검색 테스트
    print_section_header("검색 테스트")
    
    query = "딥러닝의 주요 아키텍처는 무엇인가요?"
    print(f"\n쿼리: '{query}'")
    
    results = rag.search(query, n_results=3)
    
    print("\n검색 결과:")
    for i, result in enumerate(results, 1):
        # 점수 해석 추가
        score_desc = ""
        if result.score >= 0.5:
            score_desc = " [v] 높은 관련성"
        elif result.score >= 0.3:
            score_desc = " [~] 중간 관련성"
        else:
            score_desc = " [x] 낮은 관련성"
        
        chunk_id = format_chunk_id(result.metadata)
        
        print(f"  [{i}] 점수: {result.score:.4f} (거리 기반 0~1){score_desc} | 청크 #{chunk_id}")
        
        if i == 1:  # 첫 번째만 전체 표시
            print(f"  {'─'*50}")
            print(format_chunk(result.content))
            print(f"  {'─'*50}")
        else:
            preview = result.content.replace('\n', ' ')[:80]
            print(f"      {preview}...")
    
    # 점수 해석 가이드
    print("\n" + "─"*60)
    print("[INFO] 점수 해석 가이드:")
    print("─"*60)
    print("  [계산 방법]")
    print("  * ChromaDB는 L2 거리를 반환 (0 ~ ∞, 작을수록 유사)")
    print("  * 점수 = 1/(1+거리)로 0~1 범위로 변환")
    print("  * ⚠️ Lab 1의 코사인 유사도와는 다른 개념입니다!")
    print()
    print("  ⚠️ [중요] 아래 점수 구간은 '이 실습 데이터셋' 기준 예시입니다!")
    print("     실제 서비스에서는 반드시 분포 히스토그램으로 임계값을 정해야 합니다.")
    print()
    print("  [해석 기준]")
    print("  * 0.5 ~ 1.0: 높은 관련성 [v] - L2 거리 < 1.0")
    print("  * 0.3 ~ 0.5: 중간 관련성 [~] - L2 거리 1.0~2.3")
    print("  * 0.0 ~ 0.3: 낮은 관련성 [x] - L2 거리 > 2.3")
    print()
    print("  [Lab 간 비교]")
    print("  * Lab 1: 코사인 유사도 = dot(A,B)/(||A||×||B||) → 방향 비교")
    print("  * Lab 2~3: L2 거리 변환 = 1/(1+||A-B||) → 거리 기반 점수")
    
    # 검색 품질 평가 (Recall@K)
    print_section_header("검색 품질 평가: Recall@K", "[EVAL]")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] Recall@K 란?                                    │
  │  ─────────────────────────────────────────────────────  │
  │  상위 K개 검색 결과 중 정답이 몇 개 포함되었는지 측정    │
  │                                                         │
  │  공식: Recall@K = (K개 중 정답 수) / (전체 정답 수)      │
  │                                                         │
  │  예시:                                                  │
  │  * 정답 청크: [1, 3, 5]                                 │
  │  * 검색 결과 상위 5개: [1, 2, 3, 4, 6]                  │
  │  * Recall@5 = 2/3 = 0.67 (정답 3개 중 2개 찾음)         │
  │                                                         │
  │  [TIP] 실무 목표:                                       │
  │  * Recall@1 > 0.7: 첫 번째 결과가 대체로 정답           │
  │  * Recall@5 > 0.9: 상위 5개 안에 거의 모든 정답 포함    │
  └─────────────────────────────────────────────────────────┘
  
  ────────────────────────────────────────────────────────────
  [INFO] Recall vs Precision 차이
  ────────────────────────────────────────────────────────────
  * Recall@K = (K개 중 정답 수) / (전체 정답 수)
    → "놓치지 않는 능력" - 정답을 얼마나 찾았나?
  
  * Precision@K = (K개 중 정답 수) / K
    → "쓸데없는 거 안 섞는 능력" - 검색 결과가 얼마나 정확한가?
  
  [TIP] 실무 전략:
  * RAG에서는 Recall 우선! (정답을 놓치면 답변 불가)
  * Precision은 Reranker로 보정 (상위 20개 → Top-5 재정렬)
  ────────────────────────────────────────────────────────────
    """)
    
    # 가상의 정답 청크로 평가 시연
    # 실제로는 사람이 라벨링한 정답 데이터가 필요
    print("[실험] 검색 품질 평가 시연:")
    print("  * 쿼리: '딥러닝의 주요 아키텍처'")
    print("  * 가정: 청크 #0, #1이 정답이라고 가정")
    print()
    
    # 평가 실행
    recalls = rag.evaluate_retrieval(
        query=query,
        relevant_chunk_ids=[0, 1],  # 가상의 정답
        k_values=[1, 3, 5]
    )
    
    print("  평가 결과:")
    for k, score in recalls.items():
        bar_len = int(score * 20)
        bar = "=" * bar_len + "-" * (20 - bar_len)
        status = "[v]" if score >= 0.5 else "[x]"
        print(f"    {k}: [{bar}] {score:.2f} {status}")
    
    print(f"""
  [!] 주의: 이 결과는 시연용입니다!
      실제 평가에는 사람이 라벨링한 정답 데이터가 필요합니다.
      
  [CODE] 평가 코드 예시:
  ┌─────────────────────────────────────────────────────
  │ # 1. 테스트 데이터 준비 (수작업 라벨링)
  │ test_queries = [
  │     {{"query": "딥러닝이란?", "relevant_chunks": [0, 1, 5]}},
  │     {{"query": "RAG 파이프라인", "relevant_chunks": [3, 4]}},
  │ ]
  │ 
  │ # 2. 평가 실행
  │ for test in test_queries:
  │     recalls = rag.evaluate_retrieval(
  │         query=test["query"],
  │         relevant_chunk_ids=test["relevant_chunks"]
  │     )
  │     print(f"Query: {{test['query']}}")
  │     print(f"Recall@5: {{recalls['recall@5']:.2f}}")
  └─────────────────────────────────────────────────────
    """)
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] RAG 파이프라인 핵심:")
    print("="*60)
    print("  1. 문서 로드 -> 텍스트 추출")
    print("  2. 청킹 -> 검색 단위로 분할")
    print("  3. 임베딩 -> 텍스트를 벡터로 변환")
    print("  4. 인덱싱 -> Vector DB에 저장")
    print("  5. 검색 -> 쿼리와 유사한 문서 찾기 (유사도 점수 기반)")
    
    print("""
  ────────────────────────────────────────────────────────────
  [다음 단계 예고] Re-ranking
  ────────────────────────────────────────────────────────────
  현재는 "벡터 거리 순위"만 사용합니다.
  
  [!] 한계: Bi-Encoder(임베딩)는 빠르지만 정밀도가 낮음
  
  [실무 해결책] Cross-Encoder Reranker
  ┌─────────────────────────────────────────────────────
  │ 1단계: Vector 검색으로 상위 20~50개 후보 추출 (빠름)
  │ 2단계: Cross-Encoder로 쿼리-문서 쌍을 정밀 점수화
  │ 3단계: Top-5로 재정렬 → LLM에 전달
  └─────────────────────────────────────────────────────
  
  → chapter3_advanced_patterns.py에서 실습!
  ────────────────────────────────────────────────────────────
    """)


def demo_rag_comparison():
    """실습 3: RAG 있음 vs 없음 비교"""
    print("\n" + "="*80)
    print("[3] 실습 3: RAG 있음 vs 없음 비교")
    print("="*80)
    print("목표: RAG가 답변 품질에 미치는 영향 확인")
    print("핵심: 환각(Hallucination) 감소, 정확한 정보 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # RAG 시스템 초기화
    rag = RAGSystem("demo_rag_comparison")
    
    # 샘플 텍스트 추가 (중복 체크로 이미 있으면 건너뜀)
    rag.add_text(SAMPLE_TEXT, source_name="AI_가이드", check_duplicate=True)
    
    # 질문 (문서에 있는 구체적인 정보)
    question = "RAG의 필요성과 파이프라인 단계를 설명해주세요."
    
    print(f"\n[*] 질문: {question}")
    
    # RAG 없이 답변
    print_section_header("RAG 없이 답변 (LLM 지식만 사용)", "[X]")
    answer_without_rag = rag.generate_answer_without_rag(question)
    print(answer_without_rag)
    
    # Non-RAG 문제점 분석
    print("\n" + "─"*60)
    print("[!] Non-RAG 답변의 잠재적 문제점:")
    print("─"*60)
    print("  * [X] 일반적인 설명만 제공 (특정 문서의 구체적 내용 없음)")
    print("  * [X] 학습 데이터 기반 -> 최신 정보 부재 가능")
    print("  * [X] 출처 없음 -> 정보의 신뢰성 검증 불가")
    print("  * [!] 환각(Hallucination) 위험 -> 사실과 다른 정보 생성 가능")
    
    # 환각 감지 실행 (자동 검사)
    print("\n" + "─"*60)
    print("[!] 환각(Hallucination) 자동 감지 실행:")
    print("─"*60)
    
    hallucination_check = rag.check_hallucination(
        answer=answer_without_rag,
        context=""  # Non-RAG는 컨텍스트 없음
    )
    
    if hallucination_check['potential_hallucinations']:
        print(f"  [!] 잠재적 환각 {len(hallucination_check['potential_hallucinations'])}건 발견!")
        print(f"  리스크 레벨: {hallucination_check['hallucination_risk'].upper()}")
        print()
        for i, h in enumerate(hallucination_check['potential_hallucinations'], 1):
            print(f"  {i}. {h['description']}")
            print(f"     발견: '{h['found']}'")
            if 'expected' in h:
                print(f"     기대: '{h['expected']}'")
    else:
        print("  [v] 명시적인 환각 패턴은 발견되지 않았습니다.")
        print("  [!] 주의: 이것이 답변이 정확하다는 의미는 아닙니다!")
        print("      출처 없는 정보는 항상 검증이 필요합니다.")
    
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] 환각 감지 방법                                   │
  │  ─────────────────────────────────────────────────────  │
  │  현재 구현: 규칙 기반 (키워드 매칭)                       │
  │  * 장점: 빠름, 비용 없음                                 │
  │  * 단점: 제한적, 새로운 패턴 감지 어려움                 │
  │                                                         │
  │  실무 대안:                                              │
  │  * LLM-as-Judge: 다른 LLM으로 답변 검증                  │
  │  * NLI 모델: 문맥과 답변의 논리적 일관성 검사            │
  │  * 팩트 체킹 API: 외부 지식 베이스와 대조                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # RAG로 답변
    print_section_header("RAG로 답변 (문서 검색 + LLM)", "[OK]")
    result = rag.generate_answer_with_rag(question, n_results=3)
    print(result['answer'])
    
    # RAG 답변의 장점 분석
    print("\n" + "─"*60)
    print("[OK] RAG 답변의 장점:")
    print("─"*60)
    print("  * [v] 실제 문서 기반 -> 구체적이고 정확한 정보")
    print("  * [v] 출처 명시 -> 신뢰성 검증 가능")
    print("  * [v] 최신 문서 활용 가능 -> 학습 컷오프 무관")
    print("  * [v] 환각 감소 -> 문서에 있는 내용만 활용")
    
    print("\n[LIST] 참고한 문서:")
    for i, source in enumerate(result['sources'], 1):
        chunk_id = format_chunk_id(source)
        print(f"  [{i}] 청크 #{chunk_id} ({source.get('source', 'Unknown')})")
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] RAG vs Non-RAG 핵심 차이:")
    print("="*60)
    print("  - RAG 없이: LLM의 학습 데이터에만 의존 -> 오래된/부정확할 수 있음")
    print("  - RAG 사용: 실제 문서 기반 -> 정확하고 최신 정보 제공")
    print("  - 실무: 사내 문서, 제품 매뉴얼 등에 RAG 필수")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """Chapter 1 실습 실행"""
    print("\n" + "="*80)
    print("[LAB 03 - Chapter 1] RAG 기초 파이프라인")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. 청킹(Chunking) 이해하기 - 텍스트 분할 전략")
    print("  2. 기본 RAG 파이프라인 - 문서 -> 임베딩 -> 검색")
    print("  3. RAG 있음 vs 없음 비교 - 답변 품질 차이")
    
    # 샘플 PDF 확인
    sample_pdf = Path(__file__).parent / "sample.pdf"
    if sample_pdf.exists():
        print(f"\n[FILE] PDF 파일 발견: {sample_pdf.name}")
    else:
        print(f"\n[DOC] PDF 없음 -> 내장 샘플 텍스트 사용")
    
    try:
        # 1. 청킹 이해하기
        demo_chunking()
        
        # 2. 기본 RAG 파이프라인
        demo_basic_rag()
        
        # 3. RAG 비교
        demo_rag_comparison()
        
        print("\n" + "="*80)
        print("[OK] Chapter 1 완료!")
        print("="*80)
        print("\n[NEXT] 다음 단계:")
        print("   - chapter2_rag_optimization.py : RAG 최적화 (컨텍스트 관리, 압축)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

