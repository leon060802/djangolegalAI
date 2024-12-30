import torch
from transformers import AutoTokenizer, BertForMaskedLM
from .run_relm import PTuningWrapper

def load_model_and_tokenizer(model_path, pretrained_model_path, prompt_length=1):
    """載入模型與 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    base_model = BertForMaskedLM.from_pretrained(pretrained_model_path)
    model = PTuningWrapper(base_model, prompt_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return tokenizer, model

def predict_sentence(sentence, tokenizer, model, prompt_length):
    """對單句進行模型推論與修正，處理長句時進行分段"""
    max_len = 128 - 2  # 扣除特殊 token [CLS] 和 [SEP]
    tokenized = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    # 分段處理
    segments = [input_ids[i:i + max_len] for i in range(0, len(input_ids), max_len)]
    corrected_text = ""

    for segment in segments:
        inputs = tokenizer.decode(segment, skip_special_tokens=True)
        inputs = tokenizer(inputs, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        segment_ids = inputs["input_ids"]

        # 生成 prompt_mask
        prompt_mask = torch.zeros_like(segment_ids)
        prompt_mask[:, :2 * prompt_length] = 1

        # 模型推論
        with torch.no_grad():
            outputs = model(input_ids=segment_ids, 
                            attention_mask=inputs["attention_mask"], 
                            prompt_mask=prompt_mask,  
                            apply_prompt=True)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # 解碼並清理 token
        predicted_tokens = tokenizer.convert_ids_to_tokens(predictions[0])
        input_tokens = tokenizer.convert_ids_to_tokens(segment_ids[0])

        corrected_text += clean_tokens_with_numbers(input_tokens[1:], predicted_tokens[1:])

    return corrected_text

def clean_tokens_with_numbers(input_tokens, predicted_tokens):
    """清理預測的 tokens，保留數字不變，處理 ## 拼接，排除中文數字修改，跳過 UNK token"""
    chinese_numerals = set("零一二三四五六七八九十百千萬億")
    clean_text = ""
    for input_token, predicted_token in zip(input_tokens, predicted_tokens):
        if input_token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        # 跳過 UNK token
        if predicted_token == "[UNK]":
            clean_text += input_token
        # 保留數字與中文數字不變
        elif input_token in chinese_numerals or input_token.isdigit():
            clean_text += input_token
        elif predicted_token.startswith("##"):
            clean_text += predicted_token[2:]
        else:
            clean_text += predicted_token
    return clean_text

def process_article(article_text, model_path, pretrained_model_path, prompt_length=1):
    """將文章分句並進行修正，輸出完整修正後的文章"""
    tokenizer, model = load_model_and_tokenizer(model_path, pretrained_model_path, prompt_length)
    sentences = [s + "。" for s in article_text.split("。") if s]  # 分句並保留句號
    corrected_sentences = []
    for sentence in sentences:
        corrected_sentence = predict_sentence(sentence, tokenizer, model, prompt_length)
        corrected_sentences.append(corrected_sentence)
    return "".join(corrected_sentences)