# マルチLLMプロバイダー対応 Web記事推薦システム

複数のLLMプロバイダー（Ollama (ローカル)、Google Gemini、OpenAI）に対応した、Web記事推薦システムです。

## 🎯 主な特徴

- **3つのLLMプロバイダーに対応**: Ollama、Gemini、OpenAIを環境変数で簡単切り替え
- **最新モデル対応**: GPT-5.1、Gemini 3、Gemma 3など最新モデルをサポート
- **2段階推薦アーキテクチャ**: ベクトル検索 + LLM評価で高精度推薦
- **柔軟な設定**: 環境変数またはコード内で簡単に設定変更可能
- **プライバシー重視**: Ollama使用時は完全ローカル実行
- **本番環境対応**: Webアプリケーション統合、非同期処理をサポート

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
- **特徴**: 高精度、豊富なモデル、GPT-5対応
- **コスト**: 従量課金
- **速度**: 3-5秒
- **対応モデル**: gpt-5.1, gpt-4o, gpt-4-turbo, o1-preview, o3 など
- **GPT-5対応**: 新しいパラメータ仕様（`max_completion_tokens`）に自動対応

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
OPENAI_MODEL=gpt-5.1  # または gpt-4o, o1-preview など
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

## 🚀 使用方法

### 基本的な使い方（環境変数を使用）

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

# OpenAIを使用（GPT-5.1）
recommender = LocalArticleRecommenderSystem(
    llm_provider="openai",
    llm_model="gpt-5.1"  # GPT-5.1、GPT-4o、o1-previewなど
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

## 📁 ファイル構成

```
article_recommender/
├── README.md                  # このファイル
├── ARCHITECTURE.md            # アーキテクチャ設計書
├── PROJECT_SUMMARY.md         # プロジェクトサマリー
├── .env.example               # 環境変数テンプレート
├── .env                       # 環境変数設定（自分で作成）
├── article_recommender.py     # 統合推薦システム
├── llm_recommender.py         # マルチLLMプロバイダー対応推薦エンジン
│                              # - BaseLLMProvider: 抽象基底クラス
│                              # - OllamaProvider: Ollama実装
│                              # - GeminiProvider: Gemini実装
│                              # - OpenAIProvider: OpenAI実装（GPT-5対応）
├── vector_search.py           # ベクトル検索エンジン
├── sample_articles.py         # テスト用記事データ
├── requirements.txt           # 依存パッケージ
└── .gitignore                 # Git除外設定
```

## 🏗️ アーキテクチャ

### プロバイダー抽象化

本システムは抽象化されたプロバイダーアーキテクチャを採用し、複数のLLMを統一的に扱えます：

```python
# BaseLLMProvider（抽象基底クラス）
class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

# 各プロバイダーが実装
- OllamaProvider: ローカルOllama APIを使用
- GeminiProvider: Google Gemini APIを使用
- OpenAIProvider: OpenAI APIを使用（GPT-5パラメータ自動対応）
```

### GPT-5対応の実装

OpenAIProviderは、モデル名に基づいてAPIパラメータを自動的に切り替えます：

```python
# GPT-5系、o1系、o3系: max_completion_tokens を使用
if model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3"):
    params["max_completion_tokens"] = 1000
# GPT-4系以前: max_tokens を使用
else:
    params["max_tokens"] = 1000
```

## 📊 プロバイダー比較

| 項目 | Ollama (ローカル) | Google Gemini | OpenAI |
|------|-----------------|---------------|--------|
| **処理速度** | 30-60秒（CPU）<br>5-10秒（GPU） | 3-5秒 | 3-5秒 |
| **コスト** | 無料（電気代のみ） | 従量課金（無料枠あり） | 従量課金 |
| **プライバシー** | ✅ 完全ローカル | ⚠️ データ送信あり | ⚠️ データ送信あり |
| **インターネット** | ❌ 不要 | ✅ 必要 | ✅ 必要 |
| **セットアップ** | やや複雑 | 簡単（APIキーのみ） | 簡単（APIキーのみ） |
| **精度** | 中〜高 | 高 | 最高 |
| **最新モデル** | Gemma 3 | Gemini 3 | GPT-5.1, o1, o3 |
| **推奨用途** | 開発・検証<br>プライバシー重視 | 本番環境<br>コスパ重視 | 本番環境<br>精度重視 |

## APIキーの取得方法

### Google Gemini API
1. [Google AI Studio](https://makersuite.google.com/app/apikey) にアクセス
2. 「Get API Key」をクリック
3. APIキーをコピーして`.env`ファイルに設定

### OpenAI API
1. [OpenAI Platform](https://platform.openai.com/api-keys) にアクセス
2. 「Create new secret key」をクリック
3. APIキーをコピーして`.env`ファイルに設定

## 💡 実装詳細

### 環境変数による設定管理

`.env`ファイルで一元管理されたLLM設定：

```bash
# プロバイダー切り替え
LLM_PROVIDER=openai          # ollama, gemini, openai

# 各プロバイダーの設定
OLLAMA_MODEL=gemma3:4b       # Ollamaモデル
OPENAI_MODEL=gpt-5.1         # OpenAIモデル
GEMINI_API_KEY=xxx           # GeminiのAPIキー
OPENAI_API_KEY=xxx           # OpenAIのAPIキー
```

### モデル切り替えの仕組み

システムは`LLM_PROVIDER`環境変数を読み取り、適切なプロバイダーを自動選択：

```python
# 環境変数から自動読み込み
recommender = LocalGemmaRecommender()

# またはコード内で明示的に指定
recommender = LocalGemmaRecommender(
    provider="openai",
    model="gpt-5.1",
    api_key="your-key"
)
```

### プロバイダー別の特徴

**Ollama**:
- ローカルHTTP API（デフォルト: `http://localhost:11434`）
- ストリーミング無効で同期的に応答取得
- 温度パラメータ: 0.7、最大トークン数: 1000

**Gemini**:
- `google.generativeai`ライブラリ使用
- シンプルな`generate_content()`メソッド
- 自動的にJSON形式の応答をパース

**OpenAI**:
- 最新の`openai`ライブラリ（1.0+）使用
- モデル世代別パラメータ自動切り替え
- GPT-5/o1/o3系: `max_completion_tokens`
- GPT-4系以前: `max_tokens`

## 🔧 トラブルシューティング

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

Ollamaローカルモデルが不正なJSONを生成する場合があります。以下の対策があります：

1. **より大きなモデルを使用**（gemma3:12b以上）
2. **GeminiまたはOpenAIに切り替え**（JSON生成が安定、特にGPT-5.1は非常に高精度）
3. **温度パラメータを下げる**（`llm_recommender.py`内で調整可能）

### GPT-5でパラメータエラーが出る

```
Error: Unsupported parameter: 'max_tokens' is not supported
```

**対処法**: 最新版の`llm_recommender.py`はGPT-5の新パラメータに対応済みです。最新版にアップデートしてください。

### プロバイダー切り替えがうまくいかない

```bash
# 環境変数が読み込まれているか確認
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('LLM_PROVIDER'))"

# .envファイルの存在と内容を確認
cat .env
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

## 📈 バージョン履歴

### v2.0.0 (2025年11月27日)
- ✨ **マルチLLMプロバイダー対応**: Ollama、Gemini、OpenAIの3つのプロバイダーをサポート
- ✨ **GPT-5対応**: GPT-5.1、o1、o3シリーズの新パラメータに対応
- ✨ **環境変数管理**: `.env`ファイルによる柔軟な設定管理
- 🏗️ **アーキテクチャ改善**: プロバイダー抽象化による拡張性向上
- 📚 **ドキュメント充実**: 詳細な使用例とトラブルシューティング追加

### v1.0.0 (2025年11月18日)
- 初期リリース（Ollamaのみ対応）

## ✨ まとめ

本システムは、**3つのLLMプロバイダー**に対応した柔軟な記事推薦システムです：

| プロバイダー | 推奨シーン | 主な利点 |
|------------|----------|---------|
| **Ollama** | 開発・検証環境<br>プライバシー重視の用途 | ✅ 無料<br>✅ オフライン動作<br>✅ データ漏洩リスクなし |
| **Gemini** | 本番環境（コスパ重視）<br>スタートアップ | ✅ 高速<br>✅ 無料枠あり<br>✅ 簡単セットアップ |
| **OpenAI** | 本番環境（精度重視）<br>エンタープライズ | ✅ 最高精度<br>✅ GPT-5対応<br>✅ 豊富なモデル |

### 🚀 クイックスタート

1. `.env`ファイルを作成して`LLM_PROVIDER`を設定
2. 必要に応じてAPIキーを設定
3. `python article_recommender.py`を実行

環境変数を変更するだけで、簡単にプロバイダーを切り替えられます！

## ライセンス

MIT License

## 👥 貢献

Issue、Pull Requestを歓迎します！

## 📝 作成日

- 初回作成: 2025年11月18日
- マルチLLM対応: 2025年11月20日
- 最終更新: 2025年11月27日
