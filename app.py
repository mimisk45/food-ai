import os
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from PIL import Image
from ddgs import DDGS

st.set_page_config(page_title="食事栄養チェッカー", page_icon="🍽️", layout="wide")
st.title("🍽️ 食事栄養チェッカー")
st.write("写真と入力情報から栄養素を推定し、食事の改善をサポートします。")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
if not api_key:
    st.error("Gemini APIキーが見つかりません。.env または Streamlit の Secrets に GEMINI_API_KEY を設定してください。")
    st.stop()

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.1-flash-lite"
DB_PATH = Path(__file__).with_name("nutrition_history.db")
NUTRIENTS = ["calories_kcal", "protein_g", "fat_g", "carbs_g", "fiber_g"]
NUTRIENT_LABELS = {
    "calories_kcal": "カロリー (kcal)", "protein_g": "たんぱく質 (g)",
    "fat_g": "脂質 (g)", "carbs_g": "炭水化物 (g)", "fiber_g": "食物繊維 (g)"
}


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS meals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        mode TEXT NOT NULL,
        calories REAL, protein REAL, fat REAL, carbs REAL, fiber REAL,
        missing TEXT, recommendations TEXT, details TEXT
    )""")
    conn.commit()
    return conn


def save_history(mode, total, missing, recommendations, details):
    conn = db_connect()
    conn.execute(
        "INSERT INTO meals (created_at, mode, calories, protein, fat, carbs, fiber, missing, recommendations, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), mode,
         total.get("calories_kcal"), total.get("protein_g"), total.get("fat_g"),
         total.get("carbs_g"), total.get("fiber_g"), json.dumps(missing, ensure_ascii=False),
         json.dumps(recommendations, ensure_ascii=False), json.dumps(details, ensure_ascii=False))
    )
    conn.commit()
    conn.close()


def load_history(limit=30):
    conn = db_connect()
    df = pd.read_sql_query(
        "SELECT id, created_at, mode, calories, protein, fat, carbs, fiber, missing, recommendations FROM meals ORDER BY id DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def recent_history_context(days=7):
    df = load_history(100)
    if df.empty:
        return "過去の食事記録はまだありません。"
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    recent = df[df["created_at_dt"] >= cutoff]
    if recent.empty:
        return f"直近{days}日間の記録はありません。"
    missing_items = []
    for value in recent["missing"].dropna():
        try:
            missing_items.extend(json.loads(value))
        except Exception:
            pass
    counts = pd.Series([str(x) for x in missing_items]).value_counts().head(5)
    if counts.empty:
        return f"直近{days}日間の記録は{len(recent)}件ありますが、不足栄養素の記録はありません。"
    return f"直近{days}日間の記録は{len(recent)}件です。不足の可能性が繰り返し記録された項目: " + "、".join(f"{name}（{count}回）" for name, count in counts.items())


def extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def identify_foods(image):
    prompt = """画像に写っている食べ物を識別し、食べられる部分のおおよその重量を推定してください。
断定できない食品は一般的な食品名にしてください。JSONだけを返してください。
形式: {"foods":[{"name":"食品名","grams":100,"cooking":"調理方法が分かれば記載"}]}
"""
    response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, image])
    data = extract_json(response.text)
    cleaned = []
    for item in data.get("foods", []):
        name = str(item.get("name", "")).strip()
        try:
            grams = float(item.get("grams", 100))
        except (TypeError, ValueError):
            grams = 100
        if name:
            cleaned.append({"食品名": name, "重量(g)": max(1, min(grams, 5000)), "調理方法": str(item.get("cooking", "不明")),
                            "商品名": "", "ラベルkcal/100g": None, "たんぱく質g/100g": None,
                            "脂質g/100g": None, "炭水化物g/100g": None, "食物繊維g/100g": None})
    return cleaned


def search_nutrition(food_name, cooking=""):
    query = f"{food_name} {cooking} 栄養成分 100g カロリー たんぱく質 脂質 炭水化物 食物繊維"
    results = []
    try:
        with DDGS() as ddgs:
            for result in ddgs.text(query, region="jp-jp", max_results=6):
                title, url, body = result.get("title", ""), result.get("href", ""), result.get("body", "")
                if title or url or body:
                    results.append({"title": title, "url": url, "snippet": body})
    except Exception as exc:
        return [], str(exc)
    return results, None


def analyze_all_nutrition(foods_with_sources, history_context=""):
    prompt = """
あなたは栄養情報を整理するアシスタントです。以下の食品とWeb検索結果を根拠に、各食品の100gあたりの栄養値を整理してください。
ラベル入力値がある場合はその値を優先してください。検索結果に根拠がない値はnullにしてください。
食品の状態（生・茹で・焼き等）をできるだけ一致させ、公的機関・メーカー等の情報を優先してください。
100gあたりの値を返し、計算値も返してください。単位はkcalとg。JSONのみ:
{"foods":[{"name":"食品名","grams":100,"selected_source":{"title":"ページ名","url":"https://...","reason":"理由"},"nutrition_per_100g":{"calories_kcal":null,"protein_g":null,"fat_g":null,"carbs_g":null,"fiber_g":null},"calculated_nutrition":{"calories_kcal":null,"protein_g":null,"fat_g":null,"carbs_g":null,"fiber_g":null},"comparison":"検索結果の比較","confidence":"high / medium / low"}],"comment":"食事全体の短いコメント","missing":["不足している可能性のある栄養素"],"recommendations":["追加するとよい食品"]}
健康状態を断定しないでください。直近の食事履歴も参考にし、繰り返し不足が示唆される栄養素があれば、偏りを避けた食品提案に反映してください。
食事履歴の参考情報: 
""" + history_context + "\n入力データ:\n" + json.dumps(foods_with_sources, ensure_ascii=False)
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return extract_json(response.text)


def as_number(value):
    try:
        if pd.isna(value) or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rows_from_editor(editor_df):
    rows = []
    for _, row in editor_df.iterrows():
        name = str(row.get("食品名", "")).strip()
        grams = as_number(row.get("重量(g)"))
        if not name or not grams or grams <= 0:
            continue
        label_values = {
            "calories_kcal": as_number(row.get("ラベルkcal/100g")),
            "protein_g": as_number(row.get("たんぱく質g/100g")),
            "fat_g": as_number(row.get("脂質g/100g")),
            "carbs_g": as_number(row.get("炭水化物g/100g")),
            "fiber_g": as_number(row.get("食物繊維g/100g")),
        }
        rows.append({"name": name, "grams": grams, "cooking": str(row.get("調理方法", "不明") or "不明"),
                     "brand": str(row.get("商品名", "") or "").strip(), "label_values": label_values})
    return rows


def numeric_total(food_results):
    total = {key: 0.0 for key in NUTRIENTS}
    known = {key: False for key in NUTRIENTS}
    for food in food_results:
        values = food.get("calculated_nutrition", {})
        for key in NUTRIENTS:
            value = as_number(values.get(key))
            if value is not None:
                total[key] += value
                known[key] = True
    return {key: round(total[key], 2) if known[key] else None for key in NUTRIENTS}


def display_analysis(analysis, mode, details):
    food_results = analysis.get("foods", [])
    # 栄養値の合計はPython側で再計算し、AIの算術ミスを避ける
    total = numeric_total(food_results)
    st.subheader("📊 栄養分析結果")
    cols = st.columns(5)
    for col, key, label in zip(cols, NUTRIENTS, ["カロリー", "たんぱく質", "脂質", "炭水化物", "食物繊維"]):
        value = total[key]
        unit = "kcal" if key == "calories_kcal" else "g"
        col.metric(label, f"{value:.1f} {unit}" if value is not None else "不明")
    if analysis.get("comment"):
        st.write(analysis["comment"])
    table_rows = []
    for food in food_results:
        vals = food.get("calculated_nutrition", {})
        table_rows.append({"食品": food.get("name", "食品"), "量(g)": food.get("grams", "?"),
                           "kcal": vals.get("calories_kcal"), "たんぱく質(g)": vals.get("protein_g"),
                           "脂質(g)": vals.get("fat_g"), "炭水化物(g)": vals.get("carbs_g"), "食物繊維(g)": vals.get("fiber_g")})
        with st.expander(f'{food.get("name", "食品")} — {food.get("grams", "?")} g'):
            st.write("**100gあたり**")
            st.json(food.get("nutrition_per_100g", {}))
            st.write("**選択した情報源**")
            source = food.get("selected_source") or {}
            if source.get("title"):
                st.write(source["title"])
            if source.get("reason"):
                st.write(source["reason"])
            if source.get("url", "").startswith(("https://", "http://")):
                st.markdown(f'[情報源を開く]({source["url"]})')
            if food.get("comparison"):
                st.write(food["comparison"])
            st.caption(f'情報の確信度: {food.get("confidence", "不明")}')
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    missing = analysis.get("missing", []) or []
    recommendations = analysis.get("recommendations", []) or []
    if missing:
        st.subheader("🥗 補うとよさそうな栄養素")
        for item in missing:
            st.write(f"- {item}")
    if recommendations:
        st.subheader("🍎 追加するとよい食品")
        for item in recommendations:
            st.write(f"- {item}")
    if st.button("この食事を履歴に保存", key=f"save_{mode}"):
        save_history(mode, total, missing, recommendations, details)
        st.success("食事履歴に保存しました。")
    st.caption("注意: 数値は食品データや入力の正確さに依存します。医療・診断目的には使用しないでください。")


mode = st.radio("計算モード", ["ざっくり計算", "じっくり計算"], horizontal=True,
                help="ざっくり計算は写真中心、じっくり計算は重量・調理法・商品ラベルを確認して計算します。")
uploaded_file = st.file_uploader("食事の写真（任意）", type=["png", "jpg", "jpeg", "webp"])
image = None
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードした写真", use_container_width=True)
    except Exception:
        st.error("画像を読み込めませんでした。PNG、JPG、JPEG、WEBP形式を選んでください。")
        st.stop()

if mode == "ざっくり計算":
    if st.button("🔍 栄養素を分析する", type="primary", disabled=image is None):
        try:
            with st.spinner("写真から食品と量を推定しています..."):
                identified = identify_foods(image)
            if not identified:
                st.warning("写真から食品を特定できませんでした。")
                st.stop()
            with st.spinner("PythonでWeb検索し、栄養情報を集めています..."):
                input_foods = []
                for food in identified:
                    results, error = search_nutrition(food["食品名"], food.get("調理方法", ""))
                    input_foods.append({"name": food["食品名"], "grams": food["重量(g)"], "cooking": food.get("調理方法", ""), "search_results": results, "search_error": error})
            with st.spinner("栄養素を整理しています..."):
                analysis = analyze_all_nutrition(input_foods, recent_history_context())
            # AIから返った栄養値を重量に合わせてPythonで計算し直す
            for food in analysis.get("foods", []):
                per = food.get("nutrition_per_100g", {}) or {}
                grams = as_number(food.get("grams")) or 0
                food["calculated_nutrition"] = {k: round(float(per[k]) * grams / 100, 2) if as_number(per.get(k)) is not None else None for k in NUTRIENTS}
            st.session_state["last_analysis_rough"] = analysis
            st.session_state["last_details_rough"] = input_foods
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                st.error("Gemini APIの利用枠を超えたため分析できませんでした。時間をおいて再試行してください。")
            else:
                st.error(f"分析中にエラーが発生しました: {msg}")
    if st.session_state.get("last_analysis_rough"):
        display_analysis(st.session_state["last_analysis_rough"], "rough", st.session_state.get("last_details_rough", []))

else:
    st.info("写真で食材を自動入力することも、下の表に食材を手入力することもできます。重量・調理方法を確認してから分析してください。")
    if image is not None and st.button("📷 写真から食材を読み取る"):
        try:
            with st.spinner("写真から食材を読み取っています..."):
                rows = identify_foods(image)
            st.session_state["detailed_rows"] = rows
            st.success(f"{len(rows)}種類の食材を読み取りました。表で内容を修正してください。")
        except Exception as exc:
            st.error(f"写真の読み取りに失敗しました: {exc}")
    if "detailed_rows" not in st.session_state:
        st.session_state["detailed_rows"] = [{"食品名": "ご飯", "重量(g)": 150, "調理方法": "炊飯後", "商品名": "", "ラベルkcal/100g": None, "たんぱく質g/100g": None, "脂質g/100g": None, "炭水化物g/100g": None, "食物繊維g/100g": None}]

    with st.expander("食材・調味料を追加", expanded=True):
        with st.form("add_food_form", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 2])
            new_name = c1.text_input("食品名（例：鶏むね肉、サラダ油）")
            new_grams = c2.number_input("重量(g)", min_value=0.1, max_value=5000.0, value=10.0, step=1.0)
            new_cooking = c3.text_input("調理方法", placeholder="生、茹で、焼き、炒める等")
            new_brand = st.text_input("商品名（任意）")
            st.caption("商品ラベルがある場合は、下の表で100gあたりの値を入力してください。1食あたり表示なら、表示量を基準に計算できるよう後日拡張できます。")
            submitted = st.form_submit_button("食材を追加")
        if submitted:
            if new_name.strip():
                st.session_state["detailed_rows"].append({"食品名": new_name.strip(), "重量(g)": new_grams, "調理方法": new_cooking or "不明", "商品名": new_brand,
                    "ラベルkcal/100g": None, "たんぱく質g/100g": None, "脂質g/100g": None, "炭水化物g/100g": None, "食物繊維g/100g": None})
                st.rerun()
            else:
                st.warning("食品名を入力してください。")

    st.write("**食材ごとの量・調理方法・商品ラベルを確認してください**")
    editor_df = pd.DataFrame(st.session_state["detailed_rows"])
    edited_df = st.data_editor(editor_df, num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "食品名": st.column_config.TextColumn("食品名", required=True),
            "重量(g)": st.column_config.NumberColumn("重量(g)", min_value=0.1, max_value=5000.0, step=1.0, required=True),
            "調理方法": st.column_config.TextColumn("調理方法"), "商品名": st.column_config.TextColumn("商品名"),
            "ラベルkcal/100g": st.column_config.NumberColumn("kcal/100g", min_value=0.0),
            "たんぱく質g/100g": st.column_config.NumberColumn("たんぱく質g/100g", min_value=0.0),
            "脂質g/100g": st.column_config.NumberColumn("脂質g/100g", min_value=0.0),
            "炭水化物g/100g": st.column_config.NumberColumn("炭水化物g/100g", min_value=0.0),
            "食物繊維g/100g": st.column_config.NumberColumn("食物繊維g/100g", min_value=0.0),
        }, key="detailed_editor")
    st.caption("油やドレッシングも、食品として重量を追加してください。例：サラダ油 5g。ラベル値を入力した栄養素はその値を優先し、空欄の栄養素はWeb検索で補います。")

    if st.button("🧮 じっくり栄養計算", type="primary"):
        try:
            rows = build_rows_from_editor(edited_df)
            if not rows:
                st.warning("食品名と正の重量を1件以上入力してください。")
                st.stop()
            inputs = []
            with st.spinner("食品ごとの栄養情報を検索しています..."):
                for row in rows:
                    # 商品名がある場合は検索語に加える
                    search_name = f'{row["brand"]} {row["name"]}'.strip() if row["brand"] else row["name"]
                    results, error = search_nutrition(search_name, row["cooking"])
                    inputs.append({"name": row["name"], "grams": row["grams"], "cooking": row["cooking"], "brand": row["brand"],
                                   "label_values": row["label_values"], "search_results": results, "search_error": error})
            with st.spinner("情報源を比較し、栄養値を計算しています..."):
                analysis = analyze_all_nutrition(inputs, recent_history_context())
            # 入力済みのラベル値を優先し、100g値×重量/100をPythonで厳密に再計算
            by_name = {row["name"]: row for row in rows}
            for food in analysis.get("foods", []):
                row = by_name.get(food.get("name"))
                if row is None:
                    # モデルが食品名をわずかに変えた場合、順序を使って対応
                    idx = analysis.get("foods", []).index(food)
                    row = rows[min(idx, len(rows)-1)]
                per = food.get("nutrition_per_100g", {}) or {}
                label_vals = row["label_values"]
                for key in NUTRIENTS:
                    if label_vals.get(key) is not None:
                        per[key] = label_vals[key]
                food["name"] = row["name"]
                food["grams"] = row["grams"]
                food["nutrition_per_100g"] = per
                food["calculated_nutrition"] = {key: round(float(per[key]) * row["grams"] / 100, 2) if as_number(per.get(key)) is not None else None for key in NUTRIENTS}
                if row["brand"]:
                    food["product_name"] = row["brand"]
                food["cooking_method"] = row["cooking"]
            st.session_state["last_analysis_detailed"] = analysis
            st.session_state["last_details_detailed"] = inputs
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                st.error("Gemini APIの利用枠を超えたため分析できませんでした。時間をおいて再試行してください。")
            else:
                st.error(f"じっくり計算中にエラーが発生しました: {msg}")
    if st.session_state.get("last_analysis_detailed"):
        display_analysis(st.session_state["last_analysis_detailed"], "detailed", st.session_state.get("last_details_detailed", []))

with st.expander("📚 過去の食事履歴"):
    history_df = load_history(50)
    if history_df.empty:
        st.info("まだ履歴はありません。分析結果の「この食事を履歴に保存」を押すと記録されます。")
    else:
        view = history_df.rename(columns={"created_at": "日時", "mode": "モード", "calories": "カロリー(kcal)", "protein": "たんぱく質(g)", "fat": "脂質(g)", "carbs": "炭水化物(g)", "fiber": "食物繊維(g)"})
        st.dataframe(view[["日時", "モード", "カロリー(kcal)", "たんぱく質(g)", "脂質(g)", "炭水化物(g)", "食物繊維(g)"]], use_container_width=True, hide_index=True)
        st.caption("過去の不足栄養素の記録は、次回分析時の食品提案の参考に使われます。")
