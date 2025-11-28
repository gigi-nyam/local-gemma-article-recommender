"""
パースエラーを再現・確認するテストスクリプト
"""
from llm_recommender import LocalGemmaRecommender
from sample_articles import SAMPLE_ARTICLES

print("=" * 60)
print("パースエラーテスト")
print("=" * 60)

try:
    # 推薦エンジンを初期化
    recommender = LocalGemmaRecommender()
    
    # テストクエリ
    test_query = "外務省担当局長 中国側と協議し日本側の立場説明か"
    
    print(f"\nクエリ: {test_query}")
    print("-" * 60)
    
    # 候補記事（上位10件に絞る）
    candidate_articles = SAMPLE_ARTICLES[:10]
    
    # 推薦を実行
    result = recommender.recommend_articles(
        user_query=test_query,
        candidate_articles=candidate_articles,
        top_k=3
    )
    
    print("\n✓ 成功！")
    print(f"推薦記事数: {len(result.recommendations)}")
    
except Exception as e:
    print(f"\n✗ エラー発生: {e}")
    import traceback
    traceback.print_exc()
