"""
マルチLLMプロバイダー対応の記事推薦ロジック
Ollama (ローカル)、Gemini、OpenAI に対応
"""

import os
import requests
import json
import re
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from abc import ABC, abstractmethod

# .envファイルを読み込み
load_dotenv()


class ArticleRecommendation(BaseModel):
    """推薦記事の構造化出力"""
    article_id: int
    title: str
    reason: str
    clickbait_score: float  # 0-1のスコア
    read_satisfaction_score: Optional[float] = None  # 読了後の満足度予測
    continuation_intent_score: Optional[float] = None  # 次も読みたくなる度合い
    is_serendipity: bool = False  # セレンディピティ記事かどうか
    serendipity_reason: Optional[str] = None  # セレンディピティ選択理由


class RecommendationResult(BaseModel):
    """推薦結果の構造化出力"""
    recommendations: List[ArticleRecommendation]
    reasoning: str


class BaseLLMProvider(ABC):
    """LLMプロバイダーの基底クラス"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """プロンプトからテキストを生成"""
        pass


class OllamaProvider(BaseLLMProvider):
    """Ollama API プロバイダー (ローカルモデル)"""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        
        # Ollamaが利用可能か確認
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model not in models:
                    print(f"警告: {self.model}がインストールされていません")
                    print(f"実行してください: ollama pull {self.model}")
            else:
                print(f"警告: Ollama APIに接続できません({base_url})")
        except Exception as e:
            print(f"Ollamaの確認に失敗: {e}")
    
    def generate(self, prompt: str, timeout: int = 180) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1000
            }
        }
        
        response = requests.post(url, json=payload, timeout=timeout)
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API エラー: {response.status_code} - {response.text}")
        
        result = response.json()
        return result.get('response', '')


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API プロバイダー"""

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        self.api_key = api_key
        self.model = model
        
        # google-genai (新しいSDK) のインポートと初期化
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            print(f"✓ Gemini API接続成功 (モデル: {model}) [google-genai SDK]")
        except ImportError:
            raise ImportError("google-genaiパッケージが必要です: pip install google-genai")
        except Exception as e:
            print(f"警告: Gemini API初期化エラー: {e}")
    
    def generate(self, prompt: str, timeout: int = 180) -> str:
        try:
            from google.genai import types
            
            # 環境変数でthinking_levelを指定可能にする
            # GEMINI_THINKING_LEVEL=low/high で制御（Gemini 3 Pro）
            thinking_level_env = os.getenv("GEMINI_THINKING_LEVEL", "").lower()
            thinking_level = None
            if thinking_level_env in ["low", "high"]:
                thinking_level = thinking_level_env
            
            # GenerateContentConfigの設定
            if thinking_level:
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4000,
                    thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
                )
                print(f"  💭 thinking_level={thinking_level} を適用しました")
            else:
                config = types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4000
                )
                if thinking_level_env:
                    print(f"  ℹ️ GEMINI_THINKING_LEVEL={thinking_level_env} (デフォルト動作)")
                else:
                    print(f"  ℹ️ thinking_level 未設定（デフォルト動作）")
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            
            # response.text を使用（google-genai では安全に使用可能）
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API エラー: {e}")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API プロバイダー"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            print(f"✓ OpenAI API接続成功 (モデル: {model})")
        except ImportError:
            raise ImportError("openaiパッケージが必要です: pip install openai")
        except Exception as e:
            print(f"警告: OpenAI API初期化エラー: {e}")
    
    def generate(self, prompt: str, timeout: int = 180) -> str:
        try:
            # GPT-5系では max_completion_tokens、GPT-4系以前では max_tokens を使用
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "あなたは記事推薦の専門家です。"},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # モデル名でパラメータを切り替え
            if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
                params["max_completion_tokens"] = 1000
                # GPT-5系はtemperatureのデフォルト値(1.0)のみサポート
            else:
                params["max_tokens"] = 1000
                params["temperature"] = 0.7
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API エラー: {e}")


class LocalGemmaRecommender:
    """マルチLLMプロバイダー対応の記事推薦エンジン"""
    
    def __init__(
        self, 
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            provider: LLMプロバイダー ("ollama", "gemini", "openai")。Noneの場合は環境変数から取得
            model: 使用するモデル名。Noneの場合は環境変数またはデフォルト値を使用
            api_key: APIキー。Noneの場合は環境変数から取得
            base_url: Ollama APIのベースURL (Ollamaの場合のみ)
        """
        # 環境変数から設定を取得
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        
        self.provider_name = provider
        
        # プロバイダーに応じてLLMクライアントを初期化
        if provider == "ollama":
            model = model or os.getenv("OLLAMA_MODEL", "gemma3:4b")
            base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.llm_provider = OllamaProvider(model=model, base_url=base_url)
            print(f"✓ Ollamaプロバイダーを使用 (モデル: {model})")
            
        elif provider == "gemini":
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEYが設定されていません")
            model = model or os.getenv("GEMINI_MODEL", "gemini-3-pro-preview")
            self.llm_provider = GeminiProvider(api_key=api_key, model=model)
            print(f"✓ Geminiプロバイダーを使用 (モデル: {model})")
            
        elif provider == "openai":
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEYが設定されていません")
            model = model or os.getenv("OPENAI_MODEL", "gpt-5.1")
            self.llm_provider = OpenAIProvider(api_key=api_key, model=model)
            print(f"✓ OpenAIプロバイダーを使用 (モデル: {model})")
            
        else:
            raise ValueError(f"未対応のプロバイダー: {provider}")
    
    def recommend_articles(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        top_k: int = 3
    ) -> RecommendationResult:
        """
        候補記事から「ついクリックしたくなる」記事(top_k-2件)と
        セレンディピティ記事(2件)を選択
        
        Args:
            user_query: ユーザーの検索クエリ
            candidate_articles: 候補記事のリスト
            top_k: 推薦する記事数（クリック誘引: top_k-2件 + セレンディピティ: 2件）
            
        Returns:
            推薦結果
        """
        # 候補記事を絞り込む（プロンプトが長すぎるため）
        limited_candidates = candidate_articles[:min(10, len(candidate_articles))]
        
        # クリック誘引記事の件数（最低1件）
        clickbait_count = max(1, top_k - 2)
        
        # プロンプトを構築（クリック誘引記事用）
        prompt = self._build_prompt(user_query, limited_candidates, clickbait_count)
        
        # LLM APIで推論を実行（クリック誘引記事）
        try:
            response_text = self.llm_provider.generate(prompt)
            
            # レスポンスをパース
            clickbait_result = self._parse_response(response_text, limited_candidates)
            
            # 選択されたクリック誘引記事のIDを取得
            selected_ids = [rec.article_id for rec in clickbait_result.recommendations]
            
            # セレンディピティ記事を選択
            serendipity_result = self._recommend_serendipity_articles(
                user_query=user_query,
                candidate_articles=limited_candidates,
                exclude_ids=selected_ids,
                count=2
            )
            
            # 結果を統合
            all_recommendations = clickbait_result.recommendations + serendipity_result.recommendations
            
            combined_reasoning = (
                f"【クリック誘引記事】{clickbait_result.reasoning}\n"
                f"【セレンディピティ記事】{serendipity_result.reasoning}"
            )
            
            return RecommendationResult(
                recommendations=all_recommendations,
                reasoning=combined_reasoning
            )
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            # フォールバック: 最初のtop_k件を返す
            return self._fallback_recommendation(limited_candidates, top_k)
    
    def _recommend_serendipity_articles(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        exclude_ids: List[int],
        count: int = 2
    ) -> RecommendationResult:
        """
        セレンディピティ性の高い記事を選択
        
        Args:
            user_query: ユーザーの検索クエリ
            candidate_articles: 候補記事のリスト
            exclude_ids: 除外する記事IDのリスト
            count: 選択する記事数
            
        Returns:
            セレンディピティ推薦結果
        """
        # 除外した候補記事リストを作成
        filtered_candidates = [
            article for article in candidate_articles
            if article['id'] not in exclude_ids
        ]
        
        if len(filtered_candidates) == 0:
            return RecommendationResult(
                recommendations=[],
                reasoning="セレンディピティ記事の候補がありません"
            )
        
        # セレンディピティ用プロンプトを構築
        prompt = self._build_serendipity_prompt(user_query, filtered_candidates, count)
        
        try:
            response_text = self.llm_provider.generate(prompt)
            result = self._parse_serendipity_response(response_text, filtered_candidates)
            return result
        except Exception as e:
            print(f"セレンディピティ記事選択でエラーが発生: {e}")
            return RecommendationResult(
                recommendations=[],
                reasoning="セレンディピティ記事の選択に失敗しました"
            )
    
    def _build_serendipity_prompt(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        count: int
    ) -> str:
        """セレンディピティ記事選択用プロンプトを構築"""
        articles_text = "\n".join([
            f"ID:{article['id']} [{article['summary']}] {article['title']}"
            for article in candidate_articles
        ])
        
        return f"""あなたは記事推薦の専門家です。

【タスク説明】
セレンディピティとは、ユーザーにとって「予期せぬ」かつ「関連性のある」発見のことです。

- 予期せぬ（Unexpectedness）：ユーザーの今読んだ記事からは、直接推奨される可能性が低いアイテムであることを意味します。
- 関連性（Relevance）：ユーザーの今読んだ記事への（暗黙的な）興味に密接に関連していることを意味します。

あるアイテムがセレンディピティであるためには、これら両方の条件を満たしている必要
があります。
必ず{count}件の記事を選択し、各記事についてセレンディピティ選択理由を明確に説明してください。

ユーザーが読んだ記事: 「{user_query}」

候補記事:
{articles_text}

以下の有効なJSON形式で回答してください:
{{
  "recommendations": [
    {{
      "article_id": 2,
      "title": "記事の要約",
      "serendipity_reason": "セレンディピティ選択理由（予期せぬ点と関連性を説明）",
      "unexpectedness_score": 0.85,
      "relevance_score": 0.75
    }}
  ],
  "reasoning": "全体的なセレンディピティ選択方針"
}}

必ず有効なJSON形式で出力してください。JSON以外のテキストは出力しないでください。"""
    
    def _parse_serendipity_response(
        self,
        response_text: str,
        candidate_articles: List[Dict]
    ) -> RecommendationResult:
        """セレンディピティレスポンスをパース"""
        try:
            # JSONブロックを抽出
            json_match = re.search(r'```json\s*\n(.+?)\n```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"recommendations".*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text.strip()
            
            json_text = json_text.strip()
            
            if not json_text.endswith('}'):
                last_brace = json_text.rfind('}')
                if last_brace > 0:
                    json_text = json_text[:last_brace + 1]
            
            json_text = re.sub(r'\}\s*\n\s*\{', '},\n{', json_text)
            
            data = json.loads(json_text)
            
            recommendations = []
            for rec in data.get('recommendations', []):
                article_id = rec.get('article_id')
                article = next((a for a in candidate_articles if a['id'] == article_id), None)
                
                if article:
                    # セレンディピティスコアを計算（予期せぬ度と関連性の調和平均）
                    unexpectedness = rec.get('unexpectedness_score', 0.5)
                    relevance = rec.get('relevance_score', 0.5)
                    serendipity_score = 2 * unexpectedness * relevance / (unexpectedness + relevance) if (unexpectedness + relevance) > 0 else 0
                    
                    recommendations.append(ArticleRecommendation(
                        article_id=article_id,
                        title=rec.get('title', article['title']),
                        reason=rec.get('serendipity_reason', ''),
                        clickbait_score=serendipity_score,
                        is_serendipity=True,
                        serendipity_reason=rec.get('serendipity_reason', '')
                    ))
            
            return RecommendationResult(
                recommendations=recommendations,
                reasoning=data.get('reasoning', '')
            )
            
        except Exception as e:
            print(f"セレンディピティレスポンスのパースに失敗: {e}")
            print(f"レスポンス内容（最初の500文字）: {response_text[:500]}")
            raise
    
    def _build_prompt(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        top_k: int
    ) -> str:
        """プロンプトを構築（PROMPT_TYPEに応じて切り替え）"""
        articles_text = "\n".join([
            f"ID:{article['id']} [{article['summary']}] {article['title']}"
            for article in candidate_articles
        ])
        
        prompt_type = os.getenv("PROMPT_TYPE", "satisfaction").lower()
        
        if prompt_type == "clickbait":
            # クリック誘引度重視のプロンプト
            return f"""あなたは記事推薦の専門家です。

ユーザークエリ: 「{user_query}」

以下の候補記事から「ついクリックしたくなる」記事を{top_k}件選んでください。
ただし、入力と似た記事や一般的な人気記事は避けて、この記事を読んだ後で気になりそうな記事に限ってください。

候補記事:
{articles_text}

以下の有効なJSON形式で回答してください。
重要: titleには記事の簡潔な要約を書いてください（元のタイトルをそのままコピーしない）。
重要: 文字列値は必ずダブルクォーテーション（"）で囲んでください。

{{
  "recommendations": [
    {{"article_id": 2, "title": "記事の要約", "reason": "選択理由", "clickbait_score": 0.85}}
  ],
  "reasoning": "選択方針"
}}

必ず有効なJSON形式で出力してください。JSON以外のテキストは出力しないでください。"""
        else:
            # 読了満足度重視のプロンプト（デフォルト）
            return f"""あなたは記事推薦の専門家です。

ユーザークエリ: 「{user_query}」

以下の候補記事から、読了後に満足度が高く、次も読みたくなる記事を{top_k}件選んでください。

評価基準:
1. clickbait_score: クリックしたくなる度合い（0-1）
2. read_satisfaction_score: 読了後の満足度予測（0-1）
   - タイトルと内容の一致度
   - 情報の深さと質
3. continuation_intent_score: 次も読みたくなる度合い（0-1）
   - 新たな疑問や興味を喚起するか
   - 関連トピックへの自然な導線

候補記事:
{articles_text}

JSON形式で回答してください:
{{
  "recommendations": [
    {{
      "article_id": 2,
      "title": "記事の要約",
      "reason": "選択理由",
      "clickbait_score": 0.85,
      "read_satisfaction_score": 0.90,
      "continuation_intent_score": 0.88
    }}
  ],
  "reasoning": "選択方針"
}}
"""
    
    def _parse_response(
        self,
        response_text: str,
        candidate_articles: List[Dict]
    ) -> RecommendationResult:
        """レスポンスをパース"""
        try:
            # JSONブロックを抽出
            json_match = re.search(r'```json\s*\n(.+?)\n```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"recommendations".*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    # 全体をJSONとして試す
                    json_text = response_text.strip()
            
            # 余計な文字を削除
            json_text = json_text.strip()
            
            # JSON内の引用符の問題を修正（"が"になっている場合など）
            # 不完全なJSONを修正
            if not json_text.endswith('}'):
                # 最後の完全なオブジェクトまで切り取る
                last_brace = json_text.rfind('}')
                if last_brace > 0:
                    json_text = json_text[:last_brace + 1]
            
            # Gemmaが配列の要素間のカンマを省略する問題を修正
            # 例: }{  →  },{  
            json_text = re.sub(r'\}\s*\n\s*\{', '},\n{', json_text)
            
            data = json.loads(json_text)
            
            # Pydanticモデルに変換
            recommendations = []
            for rec in data.get('recommendations', []):
                # 記事IDに対応する完全な情報を取得
                article_id = rec.get('article_id')
                article = next((a for a in candidate_articles if a['id'] == article_id), None)
                
                if article:
                    recommendations.append(ArticleRecommendation(
                        article_id=article_id,
                        title=rec.get('title', article['title']),
                        reason=rec.get('reason', ''),
                        clickbait_score=rec.get('clickbait_score', 0.5),
                        read_satisfaction_score=rec.get('read_satisfaction_score'),
                        continuation_intent_score=rec.get('continuation_intent_score')
                    ))
            
            return RecommendationResult(
                recommendations=recommendations,
                reasoning=data.get('reasoning', '')
            )
                
        except Exception as e:
            print(f"レスポンスのパースに失敗: {e}")
            print(f"レスポンス内容（最初の500文字）: {response_text[:500]}")
            raise
    
    def _fallback_recommendation(
        self,
        candidate_articles: List[Dict],
        top_k: int
    ) -> RecommendationResult:
        """フォールバック推薦"""
        recommendations = [
            ArticleRecommendation(
                article_id=article['id'],
                title=article['title'],
                reason="デフォルト推薦",
                clickbait_score=0.5
            )
            for article in candidate_articles[:top_k]
        ]
        
        return RecommendationResult(
            recommendations=recommendations,
            reasoning="エラーのためデフォルト推薦を使用"
        )


def demo_local_gemma_recommender():
    """マルチLLM推薦のデモンストレーション"""
    from sample_articles import SAMPLE_ARTICLES
    
    print("=" * 60)
    print("マルチLLM記事推薦デモ")
    print("=" * 60)
    print()
    
    # 推薦エンジンを初期化 (環境変数から設定を読み込み)
    recommender = LocalGemmaRecommender()
    
    # テストクエリ
    test_queries = [
        "岩手の住宅地近くでクマ2頭が連日柿の木に出没",
    ]
    
    for test_query in test_queries:
        print(f"ユーザークエリ: 「{test_query}」")
        print("-" * 60)
        print()
        
        # 候補記事
        candidate_articles = SAMPLE_ARTICLES
        
        print(f"候補記事数: {len(candidate_articles)}件（上位10件に絞り込み）")
        top_k = int(os.getenv("LLM_RECOMMENDATION_TOP_K", "5"))
        clickbait_count = max(1, top_k - 2)
        prompt_type = os.getenv("PROMPT_TYPE", "satisfaction").lower()
        if prompt_type == "clickbait":
            print(f"{recommender.provider_name}で「ついクリックしたくなる」記事を{clickbait_count}件 + セレンディピティ記事2件選択中...")
        else:
            print(f"{recommender.provider_name}で「読了満足度の高い」記事を{clickbait_count}件 + セレンディピティ記事2件選択中...")
        print()
        
        # 推薦を実行
        import time
        start_time = time.time()
        
        result = recommender.recommend_articles(
            user_query=test_query,
            candidate_articles=candidate_articles,
            top_k=int(os.getenv("LLM_RECOMMENDATION_TOP_K", "5"))
        )
        
        elapsed_time = time.time() - start_time
        
        # 結果を表示
        print("【推薦結果】")
        print(f"選択方針: {result.reasoning}")
        print()
        
        # 通常記事とセレンディピティ記事を分けて表示
        normal_recs = [rec for rec in result.recommendations if not rec.is_serendipity]
        serendipity_recs = [rec for rec in result.recommendations if rec.is_serendipity]
        
        if normal_recs:
            print("\n--- クリック誘引記事 ---")
            for i, rec in enumerate(normal_recs, 1):
                print(f"{i}. {rec.title}")
                print(f"   クリック誘引度: {rec.clickbait_score:.2f}")
                if rec.read_satisfaction_score is not None:
                    print(f"   読了満足度: {rec.read_satisfaction_score:.2f}")
                if rec.continuation_intent_score is not None:
                    print(f"   継続意向度: {rec.continuation_intent_score:.2f}")
                print(f"   選択理由: {rec.reason}")
                print()
        
        if serendipity_recs:
            print("\n--- セレンディピティ記事 ---")
            for i, rec in enumerate(serendipity_recs, 1):
                print(f"{i}. {rec.title}")
                print(f"   セレンディピティスコア: {rec.clickbait_score:.2f}")
                print(f"   セレンディピティ理由: {rec.serendipity_reason}")
                print()
        
        print(f"処理時間: {elapsed_time:.2f}秒")
        print()
        print("=" * 60)
        print()
    
    return recommender


if __name__ == "__main__":
    demo_local_gemma_recommender()
