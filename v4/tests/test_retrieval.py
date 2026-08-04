from retrieval import CharNgramRetriever, TfidfCharNgramRetriever


def test_retriever_returns_closest_question_answer():
    retriever = CharNgramRetriever([
        {"question": "오늘 날씨가 어때?", "answer": "맑아요."},
        {"question": "김치찌개 만드는 법?", "answer": "끓이면 돼요."},
    ])
    result = retriever.search("오늘 날씨 어때", top_k=1)[0]
    assert result["answer"] == "맑아요."


def test_retriever_has_no_empty_candidates():
    retriever = CharNgramRetriever([])
    assert retriever.search("질문", top_k=3) == []


def test_tfidf_retriever_can_filter_source_and_category():
    retriever = TfidfCharNgramRetriever([
        {"question": "오늘 날씨가 어때?", "answer": "맑아요.", "source": "chat", "category": "weather"},
        {"question": "오늘 날씨가 어때?", "answer": "검색 결과입니다.", "source": "qa", "category": "weather"},
    ])
    result = retriever.search("오늘 날씨 어때", source="qa", category="weather", top_k=1)[0]
    assert result["answer"] == "검색 결과입니다."
