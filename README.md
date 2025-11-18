# ローカルGemmaを使用したWeb記事推薦システム

Gemma 2モデルをローカル環境（Ollama）で実行し、Web記事推薦を行うシステムです。クラウドAPIに依存せず、完全にローカルで動作します。

## 概要

本システムは、**Embedding Vector検索**と**ローカルGemma推薦**を組み合わせた2段階推薦アーキテクチャを採用しています。

### システム構成

```
ユーザークエリ
    ↓
[フェーズ1] TF-IDFベクトル検索
    - 高速な類似度計算
    - 候補を10-100件に絞り込み
    ↓
[フェーズ2] ローカルGemma推薦
    - Ollama経由でGemma 2を実行
    - 「ついクリックしたくなる」記事を3件選択
    ↓
推薦結果
```

## 主な特徴

### ✅ 完全ローカル実行
- インターネット接続不要（モデルダウンロード後）
- APIキー不要
- プライバシー保護

### ✅ 低コスト
- クラウドAPIの使用料金なし
- 自前のハードウェアで実行

### ✅ カスタマイズ可能
- プロンプトの自由な調整
- モデルサイズの選択（2B, 9B, 27B）
- パラメータのチューニング

## 必要な環境

### ハードウェア要件

| モデル | 最小RAM | 推奨RAM | GPU |
|--------|---------|---------|-----|
| gemma3:4b | 4GB | 8GB | 不要 |
| gemma2:9b | 8GB | 16GB | 推奨 |
| gemma2:27b | 16GB | 32GB | 必須 |

### ソフトウェア要件

- Python 3.11+
- Ollama 0.12+
- Linux/macOS/Windows

## セットアップ

### 1. Ollamaのインストール

```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# https://ollama.com/download からインストーラーをダウンロード
```

### 2. Gemma 2モデルのダウンロード

```bash
# 軽量版（1Bパラメータ）
ollama pull gemma3:1b

# 中型版（4Bパラメータ）- 推奨
ollama pull gemma3:4b

# 大型版（27Bパラメータ）
ollama pull gemma3:27b
```

### 3. Pythonパッケージのインストール

```bash
# 仮想環境を作成
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
pip install requests  # Ollama API用
```

## 使用方法

### 基本的な使い方

```python
from article_recommender import LocalArticleRecommenderSystem
from sample_articles import SAMPLE_ARTICLES

# システムを初期化
recommender = LocalArticleRecommenderSystem(
    vector_search_top_k=10,
    llm_recommendation_top_k=3,
    gemma_model="gemma2:2b"
)

# 記事データを読み込み
recommender.fit(SAMPLE_ARTICLES)

# 推薦を実行
result = recommender.recommend("最新の政治ニュース")

# 結果を表示
for rec in result.recommendations:
    print(f"{rec.title} (スコア: {rec.clickbait_score})")
```

### デモの実行

```bash
# 完全なデモを実行
python article_recommender.py

# 個別コンポーネントのテスト
python llm_recommender.py  # ローカルGemma推薦のみ
python vector_search.py    # ベクトル検索のみ
```

## ファイル構成

```
article_recommender/
├── README.md                  # このファイル
├── article_recommender.py     # 統合推薦システム
├── llm_recommender.py         # ローカルGemma推薦エンジン
├── vector_search.py           # ベクトル検索エンジン
├── sample_articles.py         # テスト用記事データ
├── requirements.txt           # 依存パッケージ
└── .gitignore                 # Git除外設定
```

## パフォーマンス

### 処理時間（gemma3:4b, CPU実行）

| フェーズ | 処理時間 |
|---------|---------|
| ベクトル検索（20件） | 0.001-0.005秒 |
| Gemma推薦（10件→3件） | 30-60秒 |
| **合計** | **約30-60秒** |

### パフォーマンス改善のヒント

**1. より大きなモデルを使用（精度向上）**
```python
recommender = LocalArticleRecommenderSystem(
    gemma_model="gemma2:9b"  # より高精度
)
```

**2. 候補記事数を減らす（速度向上）**
```python
recommender = LocalArticleRecommenderSystem(
    vector_search_top_k=5,  # 10→5に削減
)
```

**3. GPU使用（大幅な高速化）**
```bash
# NVIDIA GPUがある場合、Ollamaが自動的に使用
# 処理時間が1/5〜1/10に短縮
```

## クラウド版との比較

| 項目 | ローカル版（Gemma） | クラウド版（Gemini） |
|------|-------------------|-------------------|
| 処理速度 | 30-60秒 | 7-8秒 |
| コスト | 無料（電気代のみ） | 従量課金 |
| プライバシー | 完全ローカル | データ送信あり |
| インターネット | 不要 | 必要 |
| セットアップ | やや複雑 | 簡単 |
| 精度 | 中〜高 | 高 |

## トラブルシューティング

### Ollamaに接続できない

```bash
# Ollamaサービスの状態を確認
systemctl status ollama

# Ollamaを再起動
sudo systemctl restart ollama

# 手動で起動
ollama serve
```

### JSONパースエラーが頻発する

Gemmaが不正なJSONを生成する場合があります。以下の対策があります：

1. **プロンプトを改善**（既に実装済み）
2. **より大きなモデルを使用**（gemma2:9b以上）
3. **温度パラメータを下げる**

```python
# llm_recommender.py の _call_ollama_api を編集
"options": {
    "temperature": 0.3,  # 0.7 → 0.3に変更
    "num_predict": 1000
}
```

### メモリ不足エラー

```bash
# より小さいモデルを使用
ollama pull gemma3:4b

# または候補記事数を減らす
recommender = LocalArticleRecommenderSystem(
    vector_search_top_k=5
)
```

### 処理が遅すぎる

```bash
# GPUドライバーをインストール（NVIDIA）
# Ubuntu/Debian
sudo apt install nvidia-driver-535

# Ollamaを再起動してGPUを認識
sudo systemctl restart ollama

# GPU使用を確認
nvidia-smi
```

## 実運用への展開

### Webアプリケーションとの統合

```python
from flask import Flask, request, jsonify
from article_recommender import LocalArticleRecommenderSystem

app = Flask(__name__)

# グローバルに初期化（起動時に1回だけ）
recommender = LocalArticleRecommenderSystem()
recommender.fit(load_articles_from_database())

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    query = data.get('query', '')
    
    result = recommender.recommend(query)
    
    return jsonify({
        'recommendations': [
            {
                'id': rec.article_id,
                'title': rec.title,
                'score': rec.clickbait_score,
                'reason': rec.reason
            }
            for rec in result.recommendations
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### バックグラウンドワーカーとして実行

推論時間が長いため、非同期処理が推奨されます。

```python
from celery import Celery
from article_recommender import LocalArticleRecommenderSystem

app = Celery('tasks', broker='redis://localhost:6379')

recommender = LocalArticleRecommenderSystem()

@app.task
def recommend_async(user_query):
    result = recommender.recommend(user_query)
    return {
        'recommendations': [
            {
                'title': rec.title,
                'score': rec.clickbait_score
            }
            for rec in result.recommendations
        ]
    }
```

## 今後の改善

### 短期的な改善

1. **JSONパース精度の向上**
   - より厳密な正規表現パターン
   - フォールバック処理の強化

2. **バッチ処理の実装**
   - 複数クエリの並列処理
   - スループットの向上

3. **キャッシング機能**
   - 頻繁なクエリの結果をキャッシュ
   - レスポンス時間の短縮

### 長期的な展望

1. **ファインチューニング**
   - 記事推薦に特化したモデル
   - 精度とJSON出力の安定性向上

2. **マルチモーダル対応**
   - 画像・動画の考慮
   - Gemma 2の視覚機能活用

3. **分散実行**
   - 複数サーバーでの負荷分散
   - 高可用性の実現

## 参考資料

- [Ollama公式サイト](https://ollama.com/)
- [Gemma 2モデル](https://ai.google.dev/gemma)
- [Ollama API ドキュメント](https://github.com/ollama/ollama/blob/main/docs/api.md)

## まとめ

ローカルGemma版は、**プライバシー**と**コスト**を重視する場合に最適です。処理速度はクラウド版に劣りますが、完全にローカルで動作し、APIキーや使用料金が不要です。

実運用では、非同期処理やキャッシングと組み合わせることで、実用的なレスポンス時間を実現できます。

## ライセンス

MIT License

## 作成日

2025年11月18日
