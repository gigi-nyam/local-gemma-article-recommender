"""
Embedding生成とベクトル検索の実装
TF-IDFを使用したシンプルな類似度検索
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import json
import re


class ArticleVectorSearch:
    """記事のベクトル検索を行うクラス"""
    
    def __init__(self):
        """初期化"""
        # 日本語対応のシンプルなトークナイザー
        def japanese_tokenizer(text):
            # 文字単位で分割（簡易的な方法）
            return [char for char in text if char.strip()]
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=1,
            analyzer='char',  # 文字レベルで解析
            token_pattern=None
        )
        self.article_vectors = None
        self.articles = []
        
    def fit(self, articles: List[Dict]) -> None:
        """
        記事データからEmbeddingを生成
        
        Args:
            articles: 記事データのリスト
        """
        self.articles = articles
        
        # タイトルと要約を結合してテキストを作成
        texts = [
            f"{article['title']} {article['summary']}"
            for article in articles
        ]
        
        # TF-IDFベクトル化
        self.article_vectors = self.vectorizer.fit_transform(texts)
        
        print(f"✓ {len(articles)}件の記事をベクトル化しました")
        print(f"  ベクトル次元数: {self.article_vectors.shape[1]}")
        
    def search_similar(
        self, 
        query: str, 
        top_k: int = 100,
        exclude_ids: List[int] = None
    ) -> List[Tuple[Dict, float]]:
        """
        クエリに類似した記事を検索
        
        Args:
            query: 検索クエリ
            top_k: 返す記事数
            exclude_ids: 除外する記事ID
            
        Returns:
            (記事, 類似度スコア)のタプルのリスト
        """
        if self.article_vectors is None:
            raise ValueError("先にfit()を呼び出してください")
        
        # クエリをベクトル化
        query_vector = self.vectorizer.transform([query])
        
        # コサイン類似度を計算
        similarities = cosine_similarity(query_vector, self.article_vectors)[0]
        
        # 類似度でソート
        sorted_indices = np.argsort(similarities)[::-1]
        
        # 除外IDを考慮して結果を作成
        exclude_ids = exclude_ids or []
        results = []
        
        for idx in sorted_indices:
            article = self.articles[idx]
            if article['id'] in exclude_ids:
                continue
                
            similarity = float(similarities[idx])
            results.append((article, similarity))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save_vectors(self, filepath: str) -> None:
        """ベクトルデータを保存"""
        import pickle
        
        data = {
            'vectorizer': self.vectorizer,
            'article_vectors': self.article_vectors,
            'articles': self.articles
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ ベクトルデータを保存しました: {filepath}")
    
    def load_vectors(self, filepath: str) -> None:
        """ベクトルデータを読み込み"""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.article_vectors = data['article_vectors']
        self.articles = data['articles']
        
        print(f"✓ ベクトルデータを読み込みました: {filepath}")


def demo_vector_search():
    """ベクトル検索のデモンストレーション"""
    from sample_articles import SAMPLE_ARTICLES
    
    print("=" * 60)
    print("ベクトル検索デモ")
    print("=" * 60)
    print()
    
    # 検索エンジンを初期化
    search_engine = ArticleVectorSearch()
    search_engine.fit(SAMPLE_ARTICLES)
    
    print()
    
    # テストクエリ
    test_queries = [
        "政治と外交のニュース",
        "事故や事件の速報",
        "経済と株価の動向",
        "スポーツと相撲",
    ]
    
    for query in test_queries:
        print(f"クエリ: 「{query}」")
        print("-" * 60)
        
        results = search_engine.search_similar(query, top_k=5)
        
        for i, (article, score) in enumerate(results, 1):
            print(f"{i}. [{score:.3f}] {article['title']}")
            print(f"   カテゴリ: {article['category']}")
        
        print()
    
    # ベクトルデータを保存
    search_engine.save_vectors('article_vectors.pkl')
    
    return search_engine


if __name__ == "__main__":
    demo_vector_search()
