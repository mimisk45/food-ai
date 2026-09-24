import os
import json

import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image


# .envを読み込む
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# ページ設定
st.set_page_config(
    page_title="食事栄養チェッカー",
    page_icon="🍽️"
)

st.title("🍽️ 食事栄養チェッカー")

st.write(
    "食事の写真をAIに分析して、"
    "栄養素と追加するとよい食品を提案します。"
)


# APIキー確認
if not api_key:
    st.error(
        "Gemini APIキーが設定されていません。\n\n"
        ".envファイルを確認してください。"
    )
    st.stop()


# Geminiクライアント
client = genai.Client(api_key=api_key)


# 画像アップロード
uploaded_file = st.file_uploader(
    "📷 食事の写真を選択してください",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file:

    # PILで画像を読み込む
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="選択した写真",
        use_container_width=True
    )


    # 分析ボタン
    if st.button("🔍 栄養素を分析する"):

        with st.spinner("AIが食事を分析しています..."):

            try:

                prompt = """
あなたは栄養管理を手伝うAIです。

この画像に写っている食事を分析してください。

写真から分かる範囲で、以下を推定してください。

1. 写っている食品
2. 各食品のおおよその重量
3. カロリー
4. タンパク質
5. 脂質
6. 炭水化物
7. 食物繊維
8. 栄養バランスについてのコメント
9. 不足している可能性がある栄養素
10. 追加するとよい食品

重要:
写真だけでは正確な重量や栄養素は分かりません。
そのため、すべて推定値として扱ってください。

JSON形式だけで回答してください。

以下の形式を必ず使用してください。

{
  "foods": [
    {
      "name": "食品名",
      "grams": 100,
      "calories": 200,
      "protein": 10,
      "fat": 5,
      "carbs": 20,
      "fiber": 3
    }
  ],
  "total": {
    "calories": 500,
    "protein": 20,
    "fat": 10,
    "carbs": 50,
    "fiber": 5
  },
  "comment": "栄養バランスについてのコメント",
  "missing": [
    "食物繊維"
  ],
  "recommendations": [
    {
      "food": "ブロッコリー",
      "reason": "食物繊維を補える可能性があります"
    }
  ]
}
"""


                # Geminiに画像とプロンプトを送信
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=[
                        prompt,
                        image
                    ]
                )


                # AIからの回答
                result_text = response.text


                # JSON部分を取得
                result_text = result_text.strip()

                if result_text.startswith("```"):
                    result_text = result_text.replace("```json", "")
                    result_text = result_text.replace("```", "")
                    result_text = result_text.strip()


                result = json.loads(result_text)


                # 分析完了
                st.success("分析が完了しました！")


                # 食品
                st.header("🍚 認識した食品")

                for food in result["foods"]:

                    st.write(
                        f"**{food['name']}** "
                        f"約{food['grams']}g"
                    )


                # 栄養素
                st.header("📊 推定栄養素")

                total = result["total"]


                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "カロリー",
                        f"{total['calories']} kcal"
                    )

                with col2:

                    st.metric(
                        "タンパク質",
                        f"{total['protein']} g"
                    )


                col3, col4 = st.columns(2)

                with col3:

                    st.metric(
                        "脂質",
                        f"{total['fat']} g"
                    )

                with col4:

                    st.metric(
                        "炭水化物",
                        f"{total['carbs']} g"
                    )


                st.metric(
                    "食物繊維",
                    f"{total['fiber']} g"
                )


                # コメント
                st.header("💬 栄養バランス")

                st.write(
                    result["comment"]
                )


                # 不足栄養素
                st.header(
                    "⚠️ 補いたい可能性がある栄養素"
                )

                for nutrient in result["missing"]:

                    st.write(
                        f"・{nutrient}"
                    )


                # おすすめ食品
                st.header(
                    "🥗 追加するとよい食品"
                )

                for recommendation in result["recommendations"]:

                    st.write(
                        f"**{recommendation['food']}**"
                    )

                    st.write(
                        recommendation["reason"]
                    )


            except json.JSONDecodeError:

                st.error(
                    "AIから返された結果をJSONとして読み取れませんでした。"
                )

                st.write("AIの回答:")
                st.write(result_text)


            except Exception as e:

                st.error(
                    f"分析中にエラーが発生しました。\n\n{e}"
                )