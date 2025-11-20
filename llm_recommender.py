"""
ローカルGemmaによるLLM推薦ロジック（HTTP API版）
Ollama HTTP APIを使用してGemmaモデルをローカルで実行
"""

import requests
import json
import re
from typing import List, Dict
from pydantic import BaseModel


class ArticleRecommendation(BaseModel):
    """推薦記事の構造化出力"""
    article_id: int
    title: str
    reason: str
    clickbait_score: float  # 0-1のスコア


class RecommendationResult(BaseModel):
    """推薦結果の構造化出力"""
    recommendations: List[ArticleRecommendation]
    reasoning: str


class LocalGemmaRecommender:
    """ローカルGemmaを使用した記事推薦エンジン（HTTP API版）"""
    
    def __init__(
        self, 
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434"
    ):
        """
        初期化
        
        Args:
            model: 使用するOllamaモデル名（gemma3:4b推奨）
            base_url: Ollama APIのベースURL
        """
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
                print(f"警告: Ollama APIに接続できません（{base_url}）")
        except Exception as e:
            print(f"Ollamaの確認に失敗: {e}")
    
    def recommend_articles(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        top_k: int = 3
    ) -> RecommendationResult:
        """
        候補記事から「ついクリックしたくなる」記事を選択
        
        Args:
            user_query: ユーザーの検索クエリ
            candidate_articles: 候補記事のリスト
            top_k: 推薦する記事数
            
        Returns:
            推薦結果
        """
        # 候補記事を絞り込む（プロンプトが長すぎるため）
        limited_candidates = candidate_articles[:min(10, len(candidate_articles))]
        
        # プロンプトを構築
        prompt = self._build_prompt(user_query, limited_candidates, top_k)
        
        # Ollama HTTP APIでローカル推論を実行
        try:
            response_text = self._call_ollama_api(prompt)
            
            # レスポンスをパース
            result = self._parse_response(response_text, limited_candidates)
            
            return result
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            # フォールバック: 最初のtop_k件を返す
            return self._fallback_recommendation(limited_candidates, top_k)
    
    def _call_ollama_api(self, prompt: str, timeout: int = 180) -> str:
        """
        Ollama HTTP APIを呼び出してレスポンスを取得
        
        Args:
            prompt: 入力プロンプト
            timeout: タイムアウト時間（秒）
            
        Returns:
            モデルのレスポンス
        """
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
    
    def _build_prompt(
        self,
        user_query: str,
        candidate_articles: List[Dict],
        top_k: int
    ) -> str:
        """プロンプトを構築（簡潔版）"""
        articles_text = "\n".join([
            # f"ID:{article['id']} {article['title']}"
            f"ID:{article['id']} [{article['summary']}] {article['title']}"
            for article in candidate_articles
        ])
        
        prompt = f"""あなたは記事推薦の専門家です。

ユーザークエリ: 「{user_query}」

以下の候補記事から「ついクリックしたくなる」記事を{top_k}件選んでください。
ただし、入力と似た記事や一般的な人気記事は避けて、この記事を読んだ後で気になりそうな記事に限ってください。

候補記事:
{articles_text}

以下のJSON形式で回答してください。
重要: titleには記事の簡潔な要約を書いてください（元のタイトルをそのままコピーしない）。
重要: JSON内では引用符(\"や")を使わないでください。

{{
  "recommendations": [
    {{"article_id": 2, "title": "記事の要約", "reason": "選択理由", "clickbait_score": 0.85}}
  ],
  "reasoning": "選択方針"
}}

JSON以外は出力しないでください。"""
        
        return prompt
    
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
                        clickbait_score=rec.get('clickbait_score', 0.5)
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
    """ローカルGemma推薦のデモンストレーション"""
    from sample_articles import SAMPLE_ARTICLES
    
    print("=" * 60)
    print("ローカルGemma推薦デモ（Ollama HTTP API使用）")
    print("=" * 60)
    print()
    
    # 推薦エンジンを初期化
    recommender = LocalGemmaRecommender(model="gemma3:4b")
    
    # テストクエリ
    test_queries = [
        "九州場所で横綱 大の里が熊を投げ倒したという噂は嘘である。九州に熊はいないのだから",
    ]
    
    for test_query in test_queries:
        print(f"ユーザークエリ: 「{test_query}」")
        print("-" * 60)
        print()
        
        # 候補記事
        candidate_articles = SAMPLE_ARTICLES
        
        print(f"候補記事数: {len(candidate_articles)}件（上位10件に絞り込み）")
        print("ローカルGemmaで「ついクリックしたくなる」記事を3件選択中...")
        print()
        
        # 推薦を実行
        import time
        start_time = time.time()
        
        result = recommender.recommend_articles(
            user_query=test_query,
            candidate_articles=candidate_articles,
            top_k=3
        )
        
        elapsed_time = time.time() - start_time
        
        # 結果を表示
        print("【推薦結果】")
        print(f"選択方針: {result.reasoning}")
        print()
        
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec.title}")
            print(f"   クリック誘引度: {rec.clickbait_score:.2f}")
            print(f"   選択理由: {rec.reason}")
            print()
        
        print(f"処理時間: {elapsed_time:.2f}秒")
        print()
        print("=" * 60)
        print()
    
    return recommender


if __name__ == "__main__":
    demo_local_gemma_recommender()
