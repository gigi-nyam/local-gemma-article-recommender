"""
ローカルGemmaを使用した統合記事推薦システム
ベクトル検索 + ローカルGemma推薦の2段階パイプライン
"""

import time
from typing import List, Dict
from vector_search import ArticleVectorSearch
from llm_recommender import LocalGemmaRecommender, RecommendationResult


class LocalArticleRecommenderSystem:
    """ローカルGemmaを使用した記事推薦システム"""
    
    def __init__(
        self,
        vector_search_top_k: int = 100,
        llm_recommendation_top_k: int = 3,
        gemma_model: str = "gemma3:4b"
    ):
        """
        初期化
        
        Args:
            vector_search_top_k: ベクトル検索で絞り込む件数
            llm_recommendation_top_k: LLMで推薦する件数
            gemma_model: 使用するGemmaモデル
        """
        self.vector_search_top_k = vector_search_top_k
        self.llm_recommendation_top_k = llm_recommendation_top_k
        
        # コンポーネントを初期化
        self.vector_search = ArticleVectorSearch()
        self.llm_recommender = LocalGemmaRecommender(model=gemma_model)
        
        print(f"✓ ローカルGemma記事推薦システムを初期化しました")
        print(f"  ベクトル検索: 上位{vector_search_top_k}件")
        print(f"  LLM推薦: 上位{llm_recommendation_top_k}件")
        print(f"  Gemmaモデル: {gemma_model}")
    
    def fit(self, articles: List[Dict]) -> None:
        """
        記事データを読み込んでベクトル化
        
        Args:
            articles: 記事データのリスト
        """
        self.vector_search.fit(articles)
        print(f"✓ {len(articles)}件の記事をベクトル化しました")
    
    def recommend(
        self,
        user_query: str,
        exclude_ids: List[int] = None
    ) -> RecommendationResult:
        """
        2段階推薦を実行
        
        Args:
            user_query: ユーザーの検索クエリ
            exclude_ids: 除外する記事IDのリスト
            
        Returns:
            推薦結果
        """
        start_time = time.time()
        
        # フェーズ1: ベクトル検索
        print(f"\n【フェーズ1】ベクトル検索で上位{self.vector_search_top_k}件に絞り込み")
        print("-" * 60)
        
        phase1_start = time.time()
        candidates = self.vector_search.search_similar(
            query=user_query,
            top_k=self.vector_search_top_k
        )
        phase1_time = time.time() - phase1_start
        
        # 除外処理
        if exclude_ids:
            candidates = [
                (article, score)
                for article, score in candidates
                if article['id'] not in exclude_ids
            ]
        
        candidate_articles = [article for article, score in candidates]
        
        print(f"✓ {len(candidate_articles)}件の候補記事を抽出")
        print(f"  処理時間: {phase1_time:.3f}秒")
        
        # 上位5件を表示
        print("上位5件の候補:")
        for i, (article, score) in enumerate(candidates[:5], 1):
            print(f"  {i}. [{score:.3f}] {article['title']}")
        
        # フェーズ2: LLM推薦
        print(f"\n【フェーズ2】ローカルGemmaで「ついクリックしたくなる」記事を{self.llm_recommendation_top_k}件選択")
        print("-" * 60)
        
        phase2_start = time.time()
        result = self.llm_recommender.recommend_articles(
            user_query=user_query,
            candidate_articles=candidate_articles,
            top_k=self.llm_recommendation_top_k
        )
        phase2_time = time.time() - phase2_start
        
        print(f"✓ {len(result.recommendations)}件の推薦記事を選択")
        print(f"  処理時間: {phase2_time:.3f}秒")
        
        # 結果を表示
        print(f"\n【最終推薦結果】")
        print("=" * 60)
        print(f"選択方針: {result.reasoning}")
        print()
        
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec.title}")
            print(f"   クリック誘引度: {rec.clickbait_score:.2f}")
            print(f"   選択理由: {rec.reason}")
            print()
        
        total_time = time.time() - start_time
        print(f"総処理時間: {total_time:.3f}秒")
        
        return result
    
    def save_model(self, filepath: str) -> None:
        """ベクトルデータを保存"""
        self.vector_search.save_vectors(filepath)
    
    def load_model(self, filepath: str) -> None:
        """ベクトルデータを読み込み"""
        self.vector_search.load_vectors(filepath)


def demo_local_recommender_system():
    """ローカルGemma推薦システムの完全デモ"""
    from sample_articles import SAMPLE_ARTICLES
    
    print("=" * 60)
    print("ローカルGemma記事推薦システム - 完全デモ")
    print("=" * 60)
    print("システム構成:")
    print("  1. Embedding Vector検索: TF-IDFベースの類似度検索")
    print("  2. LLM推薦: ローカルGemma（Ollama）による最終選択")
    print()
    
    # システムを初期化
    recommender = LocalArticleRecommenderSystem(
        vector_search_top_k=10,  # デモ用に少なめ
        llm_recommendation_top_k=3,
        gemma_model="gemma3:4b"
    )
    
    # 記事データを読み込み
    print("=" * 60)
    print("記事データの読み込み")
    print("=" * 60)
    recommender.fit(SAMPLE_ARTICLES)
    print("✓ システムの準備が完了しました")
    
    # テストクエリ
    test_queries = [
        "最新の政治と外交のニュース",
        "事故や災害の速報情報",
    ]
    
    for query in test_queries:
        print()
        print("=" * 60)
        print(f"推薦処理: 「{query}」")
        print("=" * 60)
        
        result = recommender.recommend(query)
        
        print()
        input("次のクエリに進むにはEnterキーを押してください...")
    
    # モデルを保存
    print()
    print("=" * 60)
    print("モデルの保存")
    print("=" * 60)
    recommender.save_model("article_vectors_local.pkl")
    print("✓ ベクトルデータを保存しました: article_vectors_local.pkl")
    
    return recommender


if __name__ == "__main__":
    demo_local_recommender_system()
