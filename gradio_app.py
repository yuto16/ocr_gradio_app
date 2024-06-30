import gradio as gr
from paddleocr import PaddleOCR
import cv2
import base64
import json
import os
from openai import OpenAI
import numpy as np
import pandas as pd
from google.colab import userdata

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", userdata.get('API_KEY')))

ocr = PaddleOCR(
    use_gpu=False,
    lang = "japan",
)

def image_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bilateralFilterでedgeがクッキリ
    img = cv2.bilateralFilter(img, 9, 75, 75)
    # 大津の2値化で適切な閾値に
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def image_to_ocred_text(img):
    ocr_result = ocr.ocr(img)
    ocred_text_list = []
    for res in ocr_result[0]:
        ((x1, y1), _, _, _), (temp_text, _) = res
        temp_ocred_text = f"{x1:.0f} {y1:.0f} {temp_text}"
        ocred_text_list.append(temp_ocred_text)
    return "\n".join(ocred_text_list)

def img_to_base64(img, resize=300):
    h,w,_ = img.shape

    if h>w and h>resize:
        img = cv2.resize(img, (int(resize*w/h), resize))
    elif w>h and w>resize:
        img = cv2.resize(img, (resize, int(resize*h/w)))

    _, encoded = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(encoded).decode("utf-8")
    return img_str

def ocred_text_to_json(ocred_text, img_base64):
    schema = {
        "建物名": "string",
        "住所": "string",
        "構造": "string",
        "建築年":"integer",
        "階数":"integer",
        "総戸数":"integer",
        "面積":"float",
        "電気容量":"integer",
        "ガス":"string",
        "トイレの有無":"boolean",
        "冷暖房設備の有無":"boolean",
        "駐車場の有無":"boolean",

    }

    prompt = f"""
    あなたは不動産の専門家で、賃貸借契約書の情報を綺麗に整理します。
    以下に、OCRしたテキストとBase64 foramtの画像があります。ここからoutput schemaの情報を抽出してjson形式で出力してください。
    設備の有無の情報はBase64 stringを優先して、テキスト情報はOCRテキストを使ってください。

    # knowledge
     - 建物名は○○ビル、○○ハイツ、○○タワーなどが多いです。
     - 住所は日本の住所です。
     - 構造はSRC造(鉄骨鉄筋コンクリート)、RC造(鉄筋コンクリート)、S造(鉄骨)、木造の4種類です。
     - 剣築年は建物が建設された日です。契約期間とは関係ありません。1970年から2024年が多いです。
     - 階数は建物の階数です。1から40を超えるものまであります。
     - 総戸数の単位は戸です。1から100を超えるものまであります。
     - 面積の単位はm^2です。
     - 電気容量は30アンペアや40アンペアなどの数字です。
     - ガスは日本では都市ガスかプロパンガスの2種類です。

    # Base64 image string
    {img_base64}

    # OCR text
    {ocred_text}

    # output schema
    {schema}
    """

    response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.0,
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
        ],
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=300,
    )
    # return response
    tmp_json = json.loads(response.choices[0].message.content)
    return tmp_json

def result_json_to_message_df(rj):
    type_list = []
    message_list = []
    row_data_list = []

    try:
        rj["建築年"] = int(rj["建築年"])
        if (2024-rj["建築年"])<=10:
            type_list.append("Good")
            message_list.append("10年以内の築浅の物件です！")
            row_data_list.append(f"築年数:{rj['建築年']}")
        elif (2024-rj["建築年"])>=30:
            type_list.append("Warning")
            message_list.append("築30年以上の物件です。修繕などがしっかりされているか注意しましょう。")
            row_data_list.append(f"築年数:{rj['建築年']}")
    except:
        pass

    try:
        rj["総戸数"] = int(rj["総戸数"])
        if rj["総戸数"]>=100:
            type_list.append("Good")
            message_list.append("総戸数100戸以上です！安心ですね。")
            row_data_list.append(f"総戸数:{rj['総戸数']}")
        elif rj["総戸数"]>=30:
            type_list.append("Warning")
            message_list.append("総戸数が30戸未満です。管理がしっかり行われているか注意しましょう。")
            row_data_list.append(f"総戸数:{rj['総戸数']}")
    except:
        pass

    try:
        if rj["ガス"]=="都市ガス":
            type_list.append("Good")
            message_list.append("都市ガスは利便性が高いです！")
            row_data_list.append(f"ガス:{rj['ガス']}")
        elif rj["ガス"]=="プロパンガス":
            type_list.append("Warning")
            message_list.append("プロパンガスはガス代が高くなる傾向があります。注意しましょう。")
            row_data_list.append(f"ガス:{rj['ガス']}")
    except:
        pass

    tmp_df = pd.DataFrame({
            "Type":type_list,
            "Message":message_list,
            "Row Data":row_data_list,
        })

    return tmp_df

def main(img):
    # img = image_preprocess(img)
    ocred_text = image_to_ocred_text(img)
    img_base64 = img_to_base64(img)
    result_json = ocred_text_to_json(ocred_text, img_base64)

    output_df = result_json_to_message_df(result_json)
    return output_df, result_json, ocred_text

# Webアプリを作成
app = gr.Interface(
    title="Cameda OCR app",
    description="This app extracts structured data from document and review the contents.",
    article="Ref: https://github.com/yuto16",
    fn=main,
    inputs="image",
    outputs=[
        gr.Dataframe(headers=["Type", "Message", "Row Data"], datatype=["str", "str", "str"], col_count=(3, "fixed")),
        "text",
        "text",
    ],
    allow_flagging='never'
)    #,input_size=input_size, output_size=output_size

app.launch(show_error=True, share=True)
