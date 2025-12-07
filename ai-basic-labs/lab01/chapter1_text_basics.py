"""
[Chapter 1] 텍스트 이해의 기초
- 실습 1: tiktoken으로 토큰 이해하기
- 실습 2: NLTK 전처리 파이프라인

학습 목표:
• GPT가 텍스트를 어떻게 토큰으로 분해하는지 이해
• 토큰화, 불용어 제거, 표제어 추출의 개념 파악
• 언제 전처리가 필요하고 불필요한지 구분

실행:
  python chapter1_text_basics.py
"""

import sys
from pathlib import Path
from typing import List
import tiktoken
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from dotenv import load_dotenv
import ssl

# SSL 인증서 검증 비활성화 (NLTK 다운로드용)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import
sys.path.insert(0, str(project_root))
from utils import print_section_header, print_subsection, print_key_points


# ============================================================================
# NLTK 데이터 다운로드
# ============================================================================

def download_nltk_data():
    """필요한 NLTK 데이터 다운로드"""
    print("\n[INFO] NLTK 데이터 확인 중...")
    
    resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
    ]
    
    download_needed = False
    
    for path, name in resources:
        try:
            nltk.data.find(path)
            print(f"  [OK] '{name}' 이미 설치됨")
        except LookupError:
            print(f"  [~] '{name}' 설치 확인 중...")
            download_needed = True
            try:
                result = nltk.download(name, quiet=True)
                if result:
                    print(f"  [OK] '{name}' 설치 완료")
                else:
                    print(f"  [OK] '{name}' 이미 최신 상태")
            except Exception as e:
                print(f"  [X] '{name}' 설치 실패: {e}")
    
    if not download_needed:
        print("\n[OK] 모든 NLTK 데이터가 이미 준비되어 있습니다!")
    else:
        print("\n[OK] NLTK 데이터 다운로드 완료!")


# ============================================================================
# 실습 1: tiktoken으로 토큰 이해하기
# ============================================================================

def count_tokens_with_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """tiktoken을 사용하여 텍스트의 토큰 수를 계산"""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def demo_tiktoken():
    """실습 1: tiktoken으로 토큰 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: tiktoken으로 토큰 이해하기")
    print("="*80)
    print("목표: GPT가 텍스트를 어떻게 토큰으로 분해하는지 이해")
    print("핵심: 토큰 != 단어, 한글은 영어보다 더 많은 토큰 사용")
    
    # 토큰이란 무엇인가?
    print_section_header("토큰(Token)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 토큰의 개념                                       │
  │  ─────────────────────────────────────────────────────  │
  │  • GPT는 텍스트를 '토큰' 단위로 처리합니다               │
  │  • 토큰 != 단어 (단어보다 작거나 클 수 있음)             │
  │  • 영어: 1 단어 = 1~2 토큰                               │
  │  • 한글: 1 글자 = 1.5~3 토큰 (바이트 단위 분해)          │
  │                                                         │
  │  왜 중요한가?                                            │
  │  • API 비용이 토큰 단위로 계산됨                         │
  │  • 컨텍스트 윈도우 제한이 토큰 기준                      │
  │  • 예: GPT-4 Turbo = 128K 토큰 제한                     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    texts = [
        "Hello, how are you?",
        "안녕하세요, 반갑습니다!",
        "This is a longer sentence with more words to demonstrate token counting.",
        "AI와 머신러닝은 현대 기술의 핵심입니다."
    ]
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    print_section_header("영어 vs 한글 토큰 비교", "[CMP]")
    
    for text in texts:
        token_count = count_tokens_with_tiktoken(text)
        char_count = len(text)
        chars_per_token = char_count / token_count
        
        # 효율성 해석
        if chars_per_token >= 4.0:
            efficiency = "매우 효율적"
        elif chars_per_token >= 2.5:
            efficiency = "효율적"
        elif chars_per_token >= 1.5:
            efficiency = "보통"
        else:
            efficiency = "비효율적 (토큰 많이 소모)"
        
        print(f"\n{'─'*60}")
        print(f"텍스트: {text}")
        print(f"문자 수: {char_count}자 | 토큰 수: {token_count}개")
        print(f"토큰당 문자 수: {chars_per_token:.2f}자/토큰 → {efficiency}")
        
        # 실제 토큰 ID 확인
        tokens = encoding.encode(text)
        print(f"\n토큰 ID: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # 개별 토큰 분석
        print(f"\n토큰 분석:")
        for i, token_id in enumerate(tokens[:8]):
            decoded = encoding.decode([token_id])
            byte_repr = encoding.decode_single_token_bytes(token_id)
            
            if decoded.isprintable() and not any(ord(c) > 127 for c in decoded):
                display = f"'{decoded}'"
            elif all(ord(c) > 127 for c in decoded) and decoded.isprintable():
                display = f"'{decoded}'"
            else:
                display = f"<bytes: {byte_repr.hex()}>"
            
            print(f"  [{i+1}] ID:{token_id:6d} | {display:20s} | raw: {byte_repr}")
        
        if len(tokens) > 8:
            print(f"  ... (나머지 {len(tokens) - 8}개 토큰 생략)")
        
        # 한글 토큰화 상세 설명
        if any(ord(c) > 127 for c in text):
            original_bytes = text.encode('utf-8')
            print(f"\n  [!] 한글 토큰화 상세 설명:")
            print(f"     원본 UTF-8 바이트: {original_bytes[:30]}{'...' if len(original_bytes) > 30 else ''}")
            print(f"     총 {len(original_bytes)} 바이트 (한글 1글자 = 3바이트)")
            print(f"")
            print(f"     BPE 토큰화 과정:")
            print(f"     - BPE는 바이트 레벨에서 자주 등장하는 패턴을 학습")
            print(f"     - 한글은 영어보다 학습 빈도가 낮아 더 작은 조각으로 분해")
            print(f"     - 예: '안'(U+C548) = b'\\xec\\x95\\x88' → 2개 토큰으로 분해될 수 있음")
            print(f"")
            print(f"     [OK] 모든 토큰의 바이트를 연결하면 원본 완벽 복원!")
            print(f"     [!] 개별 토큰은 유효한 문자가 아닐 수 있음 (정상)")
    
    # 핵심 포인트
    print_key_points([
        "- tiktoken: OpenAI 공식 토큰 계산 라이브러리",
        "- 모델마다 다른 인코더 사용 (gpt-3.5-turbo, gpt-4 등)",
        "- 한글은 영어보다 2~3배 더 많은 토큰 소모",
        "- API 비용 추정: 1K 토큰 = $0.001~0.01 (모델별 상이)",
        "- 실무 팁: 긴 한글 문서는 토큰 비용 미리 계산!"
    ], "tiktoken 핵심 포인트")


# ============================================================================
# 실습 2: NLTK 전처리 파이프라인
# ============================================================================

class TextPreprocessor:
    """텍스트 전처리 파이프라인"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """불용어 제거"""
        return [token for token in tokens if token not in self.stop_words]
    
    def get_wordnet_pos(self, treebank_tag: str):
        """Penn Treebank 품사 태그를 WordNet 품사로 변환"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """표제어 추출 (품사 태깅 포함)"""
        pos_tags = pos_tag(tokens)
        lemmatized = []
        for word, pos in pos_tags:
            wordnet_pos = self.get_wordnet_pos(pos)
            lemma = self.lemmatizer.lemmatize(word, pos=wordnet_pos)
            lemmatized.append(lemma)
        return lemmatized
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True) -> List[str]:
        """전체 전처리 파이프라인 실행"""
        tokens = self.tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        if lemmatize:
            tokens = self.lemmatize(tokens)
        return tokens


def demo_preprocessing():
    """실습 2: NLTK 전처리 파이프라인"""
    print("\n" + "="*80)
    print("[2] 실습 2: NLTK 전처리 파이프라인")
    print("="*80)
    print("목표: 텍스트 정규화의 필요성과 방법 이해")
    print("핵심: 토큰화 -> 정규화 -> 불용어 제거 -> 표제어 추출")
    
    # 전처리란?
    print_section_header("텍스트 전처리란?", "[DOC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 왜 전처리가 필요한가?                             │
  │  ─────────────────────────────────────────────────────  │
  │  • "Running", "runs", "ran" -> 모두 "run"의 변형         │
  │  • "the", "is", "a" -> 의미 없는 단어 (불용어)           │
  │  • 대소문자 통일 -> "AI" = "ai" = "Ai"                   │
  │                                                         │
  │  전처리 없이 검색하면?                                   │
  │  • "cats" 검색 시 "cat" 문서 놓침                        │
  │  • "THE CAT" vs "the cat" 다르게 인식                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    preprocessor = TextPreprocessor()
    
    text = "The cats are running quickly through the beautiful gardens and jumping over fences."
    
    print_section_header("단계별 전처리 과정", "[STEP]")
    print(f"\n원본 텍스트: {text}")
    
    # 1단계: 토큰화
    print_subsection("1단계: 토큰화 (Tokenization)")
    tokens = preprocessor.tokenize(text)
    print(f"  결과: {tokens}")
    print(f"  설명: 문장을 단어 단위로 분리, 소문자 변환")
    
    # 2단계: 알파벳만 남기기
    print_subsection("2단계: 알파벳 필터링")
    alpha_tokens = [token for token in tokens if token.isalpha()]
    print(f"  결과: {alpha_tokens}")
    print(f"  설명: 구두점(., !) 제거")
    
    # 3단계: 불용어 제거
    print_subsection("3단계: 불용어 제거 (Stopword Removal)")
    no_stop = preprocessor.remove_stopwords(alpha_tokens)
    removed = [t for t in alpha_tokens if t not in no_stop]
    print(f"  결과: {no_stop}")
    print(f"  제거됨: {removed}")
    print(f"  설명: 'the', 'are', 'and' 등 의미 없는 단어 제거")
    
    # 불용어 리스트 안내
    stop_words_sample = sorted(list(preprocessor.stop_words))[:15]
    print(f"\n  [INFO] 영어 불용어 (총 {len(preprocessor.stop_words)}개):")
    print(f"  예시: {stop_words_sample}...")
    
    # 4단계: 표제어 추출
    print_subsection("4단계: 표제어 추출 (Lemmatization + POS 태깅)")
    
    pos_tags = pos_tag(no_stop)
    print(f"  품사 태깅: {pos_tags}")
    print(f"""
  Penn Treebank 품사 태그 설명:
    • NN/NNS  = 명사 단수/복수 (Noun)
    • VB/VBG  = 동사 기본형/현재분사 (Verb)
    • VBD/VBN = 동사 과거형/과거분사
    • JJ      = 형용사 (Adjective)
    • RB      = 부사 (Adverb)""")
    
    lemmatized = preprocessor.lemmatize(no_stop)
    
    # 변화된 단어 강조
    changes = []
    for orig, lem in zip(no_stop, lemmatized):
        if orig != lem:
            changes.append(f"'{orig}' -> '{lem}'")
    
    print(f"\n  결과: {lemmatized}")
    if changes:
        print(f"  변환됨: {', '.join(changes)}")
    print(f"  설명: 품사에 따라 기본형으로 변환")
    
    # Lemmatization 한계 테스트
    print_subsection("Lemmatization 한계 테스트")
    print("  [실험] 비교급/최상급/불규칙 변화 단어:\n")
    
    test_words = [
        ("better", "good", "비교급 → 원급"),
        ("best", "good", "최상급 → 원급"),
        ("running", "run", "현재분사 → 기본형"),
        ("went", "go", "불규칙 과거 → 기본형"),
    ]
    
    print(f"  {'원본':<12} {'기대값':<10} {'실제결과':<12} {'성공여부':<8}")
    print(f"  {'─'*50}")
    
    lemmatizer = preprocessor.lemmatizer
    for word, expected, description in test_words:
        pos_tags = pos_tag([word])
        wordnet_pos = preprocessor.get_wordnet_pos(pos_tags[0][1])
        result = lemmatizer.lemmatize(word, pos=wordnet_pos)
        status = "[v]" if result == expected else "[x]"
        print(f"  {word:<12} {expected:<10} {result:<12} {status:<8}")
    
    # 전체 파이프라인 결과
    print_subsection("전체 파이프라인 결과")
    result = preprocessor.preprocess(text)
    print(f"  원본: {text}")
    print(f"  결과: {result}")
    print(f"  토큰 수: {len(text.split())} -> {len(result)}")
    
    # 중요 주의사항
    print_section_header("⚠️ 중요: 검색 방식별 전처리 필요성", "[WARN]")
    print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  [!] 초보자가 자주 혼동하는 핵심 포인트!                         │
  │  ─────────────────────────────────────────────────────────────  │
  │                                                                 │
  │  BM25/키워드 검색      │  임베딩 기반 검색 (Semantic)           │
  │  ─────────────────────┼────────────────────────────────────   │
  │  전처리 필수! ✓        │  전처리 불필요 (오히려 해로움!) ✗      │
  │                        │                                        │
  │  이유:                 │  이유:                                 │
  │  • 정확한 단어 매칭    │  • 임베딩 모델이 문맥 파악             │
  │  • 불용어가 노이즈     │  • 원문 그대로가 의미 보존             │
  │                        │  • 전처리 시 의미 손실!                │
  │                                                                 │
  │  [결론]                                                         │
  │  • BM25/TF-IDF → 전처리 필수 (lab03 Hybrid 검색)               │
  │  • OpenAI 임베딩 → 전처리 하지 마세요!                         │
  └─────────────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 토큰화: 텍스트를 의미 단위로 분리",
        "- 불용어 제거: 의미 없는 고빈도 단어 제거",
        "- 표제어 추출: 단어를 기본형으로 변환 (품사 태깅 필수!)",
        "- BM25엔 필수, 임베딩엔 불필요 (중요!)",
        "- 용도: 키워드 추출, BM25 검색, 텍스트 분석"
    ], "전처리 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """챕터 1 실행"""
    print("\n" + "="*80)
    print("[Chapter 1] 텍스트 이해의 기초")
    print("="*80)
    print("\n학습 목표:")
    print("  • GPT가 텍스트를 어떻게 토큰으로 분해하는지 이해")
    print("  • 텍스트 전처리 파이프라인의 각 단계 파악")
    print("  • 언제 전처리가 필요하고 불필요한지 구분")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    try:
        # 실습 1: tiktoken
        demo_tiktoken()
        
        # 실습 2: 전처리
        demo_preprocessing()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] Chapter 1 완료!")
        print("="*80)
        
        print("\n[요약]")
        print("  • 토큰: GPT의 처리 단위, 비용 계산 기준")
        print("  • 전처리: BM25엔 필수, 임베딩엔 불필요! ⚠️")
        print("  • 다음: Chapter 2에서 임베딩 생성과 검색 실습")
        
        print("\n[다음 단계]")
        print("  python chapter2_embedding_core.py")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

