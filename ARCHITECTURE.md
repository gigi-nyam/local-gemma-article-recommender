# システムアーキテクチャ設計書

## 概要

本ドキュメントでは、Web記事推薦システムの詳細なアーキテクチャと設計思想について説明します。

## 設計思想

### なぜ2段階推薦なのか

**課題:**
- 大量の記事（数万〜数百万件）から直接LLMで選択するのは非効率
- LLMのコンテキスト長には制限がある
- コストと処理時間のバランスが重要

**解決策:**
1. **第1段階（ベクトル検索）**: 高速に関連記事を絞り込み
2. **第2段階（LLM評価）**: 人間の感覚に近い高度な評価

この2段階アプローチにより、**精度とパフォーマンスの両立**を実現しています。

## システム構成図

```
┌─────────────────────────────────────────────────────────────┐
│                      ユーザークエリ                           │
│                  「最新の政治と外交のニュース」                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              フェーズ1: Embedding Vector検索                  │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  記事データ   │───▶│  TF-IDF      │───▶│ コサイン     │  │
│  │  (全記事)     │    │  ベクトル化  │    │ 類似度計算   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                ↓              │
│                                         上位100件に絞り込み    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              フェーズ2: LLM推薦                               │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  候補記事     │───▶│  Gemma 3     │───▶│ 構造化出力   │  │
│  │  (100件)     │    │     4b       │    │ (JSON)       │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                ↓              │
│                              「ついクリックしたくなる」記事3件 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      推薦結果                                 │
│  - 記事タイトル                                               │
│  - クリック誘引度スコア (0-1)                                 │
│  - 選択理由                                                   │
└─────────────────────────────────────────────────────────────┘
```

## コンポーネント詳細

### 1. ArticleVectorSearch (ベクトル検索エンジン)

**責務:**
- 記事のベクトル化
- 類似度検索
- ベクトルデータの永続化

**技術選択:**

| 選択肢 | 採用理由 | 代替案 |
|--------|----------|--------|
| TF-IDF | 軽量で高速、依存が少ない | Sentence Transformers（重い） |
| 文字n-gram | 日本語に適している | 形態素解析（MeCab等が必要） |
| scikit-learn | 標準的で安定している | Faiss（大規模向け） |

**実装の詳細:**

```python
# ベクトル化の設定
TfidfVectorizer(
    max_features=1000,      # 最大1000次元
    ngram_range=(1, 3),     # 1-3文字のn-gram
    min_df=1,               # 最低出現回数
    analyzer='char'         # 文字レベル解析
)
```

**パフォーマンス特性:**
- 時間計算量: O(n) （n = 記事数）
- 空間計算量: O(n × d) （d = ベクトル次元）
- 実測: 20件で0.001秒、10万件で約0.1秒（推定）

### 2. LLMRecommender (LLM推薦エンジン)

**責務:**
- 候補記事の評価
- 最終推薦の選択
- 選択理由の生成

**技術選択:**

| 選択肢 | 採用理由 | 代替案 |
|--------|----------|--------|
| Gemma 3 | 高速で低コスト | GPT-4（高コストだが高精度） |
| OpenAI互換API | 標準的なインターフェース | 直接API呼び出し |
| Pydantic | 型安全な構造化出力 | 手動パース |

**プロンプト設計:**

プロンプトは以下の要素で構成されています：

1. **システムプロンプト**: 役割と評価基準の定義
2. **ユーザープロンプト**: 
   - ユーザークエリ
   - 候補記事リスト
   - 出力形式の指定

**評価基準:**

```
1. タイムリー性 (Timeliness)
   - 速報性があるか
   - 最新情報か

2. インパクト (Impact)
   - 驚きや意外性があるか
   - 記憶に残るか

3. 関連性 (Relevance)
   - クエリとの関連度
   - ユーザーの意図との一致

4. 感情的訴求 (Emotional Appeal)
   - 好奇心を刺激するか
   - 感情を動かすか

5. 社会的重要性 (Social Importance)
   - 多くの人に影響するか
   - 公共性があるか
```

**構造化出力:**

```python
class ArticleRecommendation(BaseModel):
    article_id: int                # 記事ID
    title: str                     # タイトル
    reason: str                    # 選択理由
    clickbait_score: float         # 0-1のスコア

class RecommendationResult(BaseModel):
    recommendations: List[ArticleRecommendation]
    reasoning: str                 # 全体的な選択方針
```

### 3. ArticleRecommenderSystem (統合システム)

**責務:**
- 2つのコンポーネントの統合
- パイプライン全体の制御
- パフォーマンス測定

**処理フロー:**

```python
def recommend(user_query: str) -> RecommendationResult:
    # フェーズ1: ベクトル検索
    candidates = vector_search.search_similar(
        query=user_query,
        top_k=100
    )
    
    # フェーズ2: LLM推薦
    result = llm_recommender.recommend_articles(
        user_query=user_query,
        candidate_articles=candidates,
        top_k=3
    )
    
    return result
```

## データフロー

### 記事データの構造

```python
{
    "id": 1,                                    # 一意のID
    "title": "記事タイトル",                     # タイトル
    "summary": "記事の要約",                     # 要約
    "category": "政治",                         # カテゴリ
    "timestamp": "2025-11-18 12:58"            # タイムスタンプ
}
```

### ベクトルデータの永続化

```python
# 保存
{
    'vectorizer': TfidfVectorizer(...),        # ベクトル化器
    'article_vectors': sparse_matrix,          # ベクトル行列
    'articles': List[Dict]                     # 記事データ
}

# pickle形式で保存
import pickle
with open('article_vectors.pkl', 'wb') as f:
    pickle.dump(data, f)
```

## スケーラビリティ

### 記事数の増加への対応

**現在の実装（〜10万件）:**
- TF-IDF + scikit-learn
- インメモリ検索

**大規模化（10万件〜）:**
- Faiss（Facebook AI Similarity Search）
- ベクトルDBの導入（Pinecone, Weaviate等）

```python
# Faissの例
import faiss

# インデックスを作成
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

# 検索
distances, indices = index.search(query_vector, k=100)
```

### リアルタイム性の向上

**キャッシング戦略:**

```python
# Redis等でキャッシュ
cache_key = f"recommend:{hash(query)}"
cached_result = redis.get(cache_key)

if cached_result:
    return cached_result
else:
    result = recommender.recommend(query)
    redis.setex(cache_key, 3600, result)  # 1時間キャッシュ
    return result
```

**非同期処理:**

```python
import asyncio

async def recommend_async(query: str):
    # ベクトル検索（高速）
    candidates = await vector_search_async(query)
    
    # LLM推薦（並列化可能）
    tasks = [
        llm_recommend_batch(candidates[i:i+10])
        for i in range(0, len(candidates), 10)
    ]
    results = await asyncio.gather(*tasks)
    
    return merge_results(results)
```

## セキュリティとプライバシー

### APIキーの管理

```python
# 環境変数から読み込み
import os
api_key = os.getenv('OPENAI_API_KEY')

# 決してハードコードしない
# api_key = "sk-..." ← NG
```

### ユーザーデータの保護

- クエリログは匿名化
- 個人情報を含む記事は除外
- GDPR/個人情報保護法への対応

## モニタリングと改善

### 重要な指標

**パフォーマンス指標:**
- レスポンス時間
- スループット（QPS）
- エラー率

**品質指標:**
- クリック率（CTR）
- 滞在時間
- ユーザー満足度

**コスト指標:**
- LLM APIコスト
- インフラコスト

### A/Bテストの設計

```python
# 実験グループの割り当て
def get_experiment_group(user_id: str) -> str:
    hash_value = hash(user_id) % 100
    if hash_value < 50:
        return "control"  # 既存アルゴリズム
    else:
        return "treatment"  # 新アルゴリズム

# 結果の記録
def log_recommendation(
    user_id: str,
    group: str,
    query: str,
    recommendations: List[Dict],
    clicked: bool
):
    # データベースに記録
    pass
```

## 今後の拡張

### 短期的な改善

1. **パーソナライゼーション**
   - ユーザーの閲覧履歴を考慮
   - 興味カテゴリの学習

2. **多様性の向上**
   - カテゴリの多様性を確保
   - フィルターバブルの回避

3. **説明可能性**
   - なぜこの記事が推薦されたのか
   - ユーザーへの透明性

### 長期的な展望

1. **マルチモーダル対応**
   - 画像・動画の考慮
   - タイトルだけでなく本文も分析

2. **リアルタイム学習**
   - ユーザーフィードバックの即座反映
   - オンライン学習の導入

3. **クロスプラットフォーム**
   - モバイルアプリ対応
   - API提供

## 参考文献

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805)
- [Recommender Systems Handbook](https://www.springer.com/gp/book/9780387858203)

## まとめ

本システムは、**高速なベクトル検索**と**高度なLLM評価**を組み合わせることで、ユーザーにとって魅力的な記事推薦を実現しています。

実運用に向けては、スケーラビリティ、パーソナライゼーション、モニタリングの強化が重要です。
