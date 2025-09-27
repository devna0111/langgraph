from ddgs import DDGS

def search_duckduckgo(query, max_results=5):
    with DDGS() as ddgs:
        results = ddgs.text(query)
        return [r["body"] for r in results]
    
if __name__ == "__main__":
    query = "9월 28일 서울 날씨"
    results = search_duckduckgo(query)
    print(len(results))
    print(results[0])
    print(results[1])
    print("2024년 정보만 반환해주는 이유는 뭔데")