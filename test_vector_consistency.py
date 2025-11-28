"""
ベクトル検索の一貫性をテストするスクリプト
"""
from article_recommender import LocalArticleRecommenderSystem
from sample_articles import SAMPLE_ARTICLES

# システムを初期化
recommender = LocalArticleRecommenderSystem(vector_search_top_k=10, llm_recommendation_top_k=3)
recommender.fit(SAMPLE_ARTICLES)

# 同じクエリで3回検索
query = '外務省担当局長 中国側と協議し日本側の立場説明か'

print(f"クエリ: {query}\n")

for i in range(3):
    print(f'=== 試行 {i+1} ===')
    candidates = recommender.vector_search.search_similar(query, top_k=10)
    article_ids = [article['id'] for article, score in candidates]
    print(f"記事ID: {article_ids}")
    
    print("上位5件:")
    for j, (article, score) in enumerate(candidates[:5], 1):
        title_short = article['title'][:40] + '...' if len(article['title']) > 40 else article['title']
        print(f"  {j}. [スコア:{score:.4f}] ID:{article['id']} {title_short}")
    print()
