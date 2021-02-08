import pandas

queries = ["machine+learning", "data+science", "kaggle"]
url = "https://www.youtube.com/results?search_query={query}&sp=CAI%253D&p={page}"

for query in queries:
    for page in range(1,101):
        urll = url.format(query=query, page=page)
        print(urll)
