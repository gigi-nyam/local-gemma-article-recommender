# マルチLLMプロバイダー対応 Web記事推薦システム

複数のLLMプロバイダー（Ollama (ローカル)、Google Gemini、OpenAI）に対応した、Web記事推薦システムです。

## 概要

本システムは、**Embedding Vector検索**と**LLM推薦**を組み合わせた2段階推薦アーキテクチャを採用しています。

### システム構成

```
ユーザークエリ
    ↓
[フェーズ1] TF-IDFベクトル検索
    - 高速な類似度計算
    - 候補を10-100件に絞り込み
    ↓
[フェーズ2] LLM推薦
    - Ollama/Gemini/OpenAIで推論
    - 「ついクリックしたくなる」記事を3件選択
    ↓
推薦結果
```

## 対応LLMプロバイダー

### 1. Ollama (ローカル実行)
- **特徴**: 完全ローカル、APIキー不要、プライバシー保護
- **コスト**: 無料（電気代のみ）
- **速度**: 30-60秒（CPU）、5-10秒（GPU）
- **対応モデル**: gemma3:1b, gemma3:4b, gemma3:12b など

### 2. Google Gemini
- **特徴**: 高速、高精度
- **コスト**: 従量課金（無料枠あり）
- **速度**: 3-5秒
- **対応モデル**: gemini-1.5-flash, gemini-1.5-pro など

### 3. OpenAI
- **特徴**: 高精度、豊富なモデル
- **コスト**: 従量課金
- **速度**: 3-5秒
- **対応モデル**: gpt-4o, gpt-4o, gpt-4 など

## セットアップ

### 1. 環境変数の設定

`.env`ファイルを作成し、使用するプロバイダーに応じて設定します。

```bash
# .envファイルの例をコピー
cp .env.example .env
```

**.env ファイルの設定例**:

```bash
# LLMプロバイダーの選択
# Options: ollama, gemini, openai
LLM_PROVIDER=ollama

# Ollama設定 (LLM_PROVIDER=ollama の場合)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b

# Gemini設定 (LLM_PROVIDER=gemini の場合)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI設定 (LLM_PROVIDER=openai の場合)
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Ollama使用時の追加セットアップ

Ollamaを使用する場合のみ、以下のセットアップが必要です。

**Ollamaのインストール**:
```bash
# Linux/macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# https://ollama.com/download からインストーラーをダウンロード
```

**モデルのダウンロード**:
```bash
# 軽量版（1Bパラメータ）
ollama pull gemma3:1b

# 中型版（4Bパラメータ）- 推奨
ollama pull gemma3:4b

# 大型版（12Bパラメータ）
ollama pull gemma3:12b
```

### 3. Pythonパッケージのインストール

```bash
# 仮想環境を作成
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存パッケージをインストール
pip install -r requirements.txt
```

**requirements.txt に含まれるパッケージ**:
- `python-dotenv`: 環境変数管理
- `google-generativeai`: Gemini API（オプション）
- `openai`: OpenAI API（オプション）
- その他の基本パッケージ

## 使用方法

### 基本的な使い方

```python
from article_recommender import LocalArticleRecommenderSystem
from sample_articles import SAMPLE_ARTICLES

# システムを初期化（環境変数から設定を読み込み）
recommender = LocalArticleRecommenderSystem(
    vector_search_top_k=10,
    llm_recommendation_top_k=3
)

# 記事データを読み込み
recommender.fit(SAMPLE_ARTICLES)

# 推薦を実行
result = recommender.recommend("最新の政治ニュース")

# 結果を表示
for rec in result.recommendations:
    print(f"{rec.title} (スコア: {rec.clickbait_score})")
```

### プロバイダーを直接指定

環境変数の代わりに、コード内で直接プロバイダーを指定することもできます。

```python
# Geminiを使用
recommender = LocalArticleRecommenderSystem(
    llm_provider="gemini",
    llm_model="gemini-1.5-flash"
)

# OpenAIを使用
recommender = LocalArticleRecommenderSystem(
    llm_provider="openai",
    llm_model="gpt-4o"
)

# Ollamaを使用
recommender = LocalArticleRecommenderSystem(
    llm_provider="ollama",
    llm_model="gemma3:4b"
)
```

### デモの実行

```bash
# 完全なデモを実行（.envファイルの設定を使用）
python article_recommender.py

# 個別コンポーネントのテスト
python llm_recommender.py  # LLM推薦のみ
python vector_search.py    # ベクトル検索のみ
```

## ファイル構成

```
article_recommender/
├── README.md                  # このファイル
├── .env.example               # 環境変数テンプレート
├── .env                       # 環境変数設定（自分で作成）
├── article_recommender.py     # 統合推薦システム
├── llm_recommender.py         # マルチLLMプロバイダー対応推薦エンジン
├── vector_search.py           # ベクトル検索エンジン
├── sample_articles.py         # テスト用記事データ
├── requirements.txt           # 依存パッケージ
└── .gitignore                 # Git除外設定
```

## プロバイダー比較

| 項目 | Ollama (ローカル) | Google Gemini | OpenAI |
|------|-----------------|---------------|--------|
| 処理速度 | 30-60秒（CPU）<br>5-10秒（GPU） | 3-5秒 | 3-5秒 |
| コスト | 無料（電気代のみ） | 従量課金（無料枠あり） | 従量課金 |
| プライバシー | 完全ローカル | データ送信あり | データ送信あり |
| インターネット | 不要 | 必要 | 必要 |
| セットアップ | やや複雑 | 簡単（APIキーのみ） | 簡単（APIキーのみ） |
| 精度 | 中〜高 | 高 | 高 |

## APIキーの取得方法

### Google Gemini API
1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. 「Get API Key」をクリック
3. APIキーをコピーして`.env`ファイルに設定

### OpenAI API
1. [OpenAI Platform](https://platform.openai.com/api-keys) にアクセス
2. 「Create new secret key」をクリック
3. APIキーをコピーして`.env`ファイルに設定

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

### APIキーエラー

```python
# エラーメッセージ例
ValueError: GEMINI_API_KEYが設定されていません
```

**対処法**: `.env`ファイルに正しいAPIキーが設定されているか確認してください。

### JSONパースエラーが頻発する

Gemmaが不正なJSONを生成する場合があります。以下の対策があります：

1. **より大きなモデルを使用**（gemma3:12b以上）
2. **GeminiまたはOpenAIに切り替え**（JSON生成が安定）

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

推論時間が長い場合、非同期処理が推奨されます。

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

## 参考資料

- [Ollama公式サイト](https://ollama.com/)
- [Google Gemini API](https://ai.google.dev/gemma)
- [OpenAI API](https://platform.openai.com/docs/)
- [Ollama API ドキュメント](https://github.com/ollama/ollama/blob/main/docs/api.md)

## まとめ

本システムは、3つのLLMプロバイダーに対応し、ニーズに応じて柔軟に選択できます：

- **Ollama**: プライバシー重視、コスト削減、オフライン動作
- **Gemini**: 高速・高精度、無料枠で開始可能
- **OpenAI**: 最高精度、豊富なモデル選択肢

環境変数を変更するだけで簡単にプロバイダーを切り替えられます。

## ライセンス

MIT License

## 作成日

2025年11月20日
