import pandas as pd
import difflib
from collections import Counter
import re  # <--- Bổ sung thư viện Regular Expression

# 1. KNOWLEDGE BASE

STOPWORDS = set("之乎者也矣焉哉以而其于於乃則爲所與")

SYNONYM_MAP = {
    "尙": "尚", "参": "參", "茂": "懋", "生": "誕", 
    "将": "領", "率": "領", "引": "領", 
    "差": "遣", "擊": "討", "攻": "討", 
    "止": "但", "唯": "惟", "以之": "由是", "端供": "認保",
    "帝": "皇帝", "曰": "為", "神": "臣", "㫖揮": "勑旨"
}

# Mapping Chinese numbers to integer values for semantic comparison
CHINESE_NUM_MAP = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '百': 100, '千': 1000, '萬': 10000
}

TIME_KEYWORDS = set("初朔春夏秋冬年月日子丑寅卯辰巳午未申酉戌亥甲乙丙丁戊己庚辛壬癸")

MULTI_CHAR_ENTITIES = [
    "鄭松", "黄廷爱", "陳登選", "阮能紹", "武惟志", "潘兼全", "范公著", "阮倦", "鄭杜",
    "尚書", "部尙書", "户部", "禮部", "兵部", "刑部", "吏部", "節制", "郡公", 
    "太傅", "太保", "太尉", "都督", "将軍", "總兵", "進士", "同進士", "進士出身", "社長", "諸營", 
    "宋朝", "明朝", "占城", "韃靼", "哀牢", "大理", 
    "清化", "乂安", "演州", "東關", "昇龍", "大羅", "諒山", "太原", "順化", "廣南", 
    "三江", "興化", "宣光", "龍州", "憑祥", "嘉遠", "交州", "兵象", "統領兵", "洞靈臺"
]

MULTI_CHAR_ENTITIES = sorted(MULTI_CHAR_ENTITIES, key=len, reverse=True)

def normalize(text):
    for k, v in SYNONYM_MAP.items():
        text = text.replace(k, v)
    return text

def get_meaningful(text):
    return [c for c in text if c not in STOPWORDS]

def parse_chinese_number(text):
    """Extract and evaluate numerical values embedded in text strings."""
    nums = []
    current_val = 0
    has_num = False
    
    for c in text:
        if c in CHINESE_NUM_MAP:
            val = CHINESE_NUM_MAP[c]
            has_num = True
            if val == 10 or val == 100 or val == 1000 or val == 10000:
                if current_val == 0:
                    current_val = 1
                current_val *= val
            else:
                current_val += val
        else:
            if has_num:
                nums.append(current_val)
                current_val = 0
                has_num = False
    if has_num:
        nums.append(current_val)
        
    # Fallback to direct digits if any Arabic numerals exist
    for c in text:
        if c.isdigit():
            nums.append(int(c))
            
    return nums

# 2. EVALUATION PIPELINE (ROBUST & STRICT)

def evaluate_pipeline(row):
    ref_raw = str(row.get('Reference', '')).replace(" ", "")
    pred_raw = str(row.get('Prediction', '')).replace(" ", "")
    
    ref = normalize(ref_raw)
    pred = normalize(pred_raw)
    
    labels = ['CORRECT', 'MISTRANSLATION', 'OMISSION', 'UNSUPPORTED_ADDITION', 'ENTITY_ERROR', 'NUMBER_OR_TIME_ERROR', 'OTHER']
    for col in labels:
        row[col] = 0
    notes = []
    
    # 0. Perfect Match
    if ref == pred:
        row['CORRECT'] = 1
        row['Notes'] = "Exact"
        return row
        
    # ==========================================
    # BƯỚC MỚI: KIỂM TRA LỖI KỸ THUẬT (OTHER)
    # ==========================================
    
    # 0.1 Lỗi Code-Switching (Phát hiện chữ Quốc ngữ / Latin)
    if re.search(r'[a-zA-Z]', pred_raw):
        row['OTHER'] = 1
        row['Notes'] = "Code-Switching"
        return row  # Thoát sớm để không bắt nhầm Omission/Addition
        
    # 0.2 Lỗi Sụp đổ vòng lặp (Infinite Looping)
    # Tìm các cụm từ (>= 2 ký tự) bị lặp lại LIÊN TIẾP từ 4 lần trở lên
    if re.search(r'(.{2,})\1{3,}', pred_raw):
        row['OTHER'] = 1
        row['Notes'] = "Infinite Looping"
        return row  # Thoát sớm để không bắt nhầm Unsupported Addition
        
    # ==========================================
        
    # 1. Advanced Number & Time Error Checking
    ref_nums = parse_chinese_number(ref)
    pred_nums = parse_chinese_number(pred)
    
    ref_time_chars = {c for c in ref if c in TIME_KEYWORDS}
    pred_time_chars = {c for c in pred if c in TIME_KEYWORDS}
    
    # Check if numerical values or crucial time markers mismatch
    if ref_nums != pred_nums or ref_time_chars != pred_time_chars:
        row['NUMBER_OR_TIME_ERROR'] = 1
        notes.append("Time/Num")
        
    # 2. Entity Error & Hallucination
    ref_temp, pred_temp = ref_raw, pred_raw
    for entity in MULTI_CHAR_ENTITIES:
        r_count = ref_temp.count(entity)
        p_count = pred_temp.count(entity)
        if r_count > p_count:
            row['ENTITY_ERROR'] = 1
            notes.append(f"-Entity({entity})")
        elif r_count < p_count:
            row['ENTITY_ERROR'] = 1
            row['UNSUPPORTED_ADDITION'] = 1
            notes.append(f"+Entity({entity})")
        
        ref_temp = ref_temp.replace(entity, '█')
        pred_temp = pred_temp.replace(entity, '█')
            
    # 3. Bag-of-Words (BoW) Analysis
    ref_mean = get_meaningful(ref)
    pred_mean = get_meaningful(pred)
    
    ref_cnt, pred_cnt = Counter(ref_mean), Counter(pred_mean)
    missing = sum((ref_cnt - pred_cnt).values())
    added = sum((pred_cnt - ref_cnt).values())
    
    len_ref = len(ref_mean)
    overlap = sum((ref_cnt & pred_cnt).values())
    coverage = overlap / len_ref if len_ref > 0 else 1
    
    omission_threshold = 2 if len_ref <= 6 else 3
    addition_threshold = 2 if len_ref <= 6 else 3
    
    # 4. Omission
    if missing >= omission_threshold:
        row['OMISSION'] = 1
        notes.append(f"Omit({missing})")
        
    # 5. Unsupported Addition
    if added >= addition_threshold and len(pred_mean) > len_ref * 1.15:
        row['UNSUPPORTED_ADDITION'] = 1
        notes.append("Add(Verbose)")
        
    # 6. Mistranslation (Strict Mutually Exclusive Fallback - 95% Coverage)
    has_specific_errors = (row['OMISSION'] == 1 or row['UNSUPPORTED_ADDITION'] == 1 or row['ENTITY_ERROR'] == 1 or row['NUMBER_OR_TIME_ERROR'] == 1)
    
    if not has_specific_errors and ref != pred:
        if coverage >= 0.95 or difflib.SequenceMatcher(None, ref, pred).ratio() >= 0.95:
            row['CORRECT'] = 1
            notes.append("Minor Var")
        else:
            row['MISTRANSLATION'] = 1
            notes.append("Mistrans")
            
    # 7. Smart Fallback Catch-all
    error_sum = sum(row[col] for col in labels if col not in ['CORRECT', 'OTHER'])
    if error_sum == 0 and row['CORRECT'] == 0:
        row['OTHER'] = 1
        notes.append("Other")
                
    row['Notes'] = " | ".join(notes)
    return row

# 3. EXECUTION

if __name__ == "__main__":
    input_file = "Error_Analysis_PreLabeled.xlsx" 
    output_file = "Error_Analysis_Labeled.xlsx"

    print("Processing data...")
    xls = pd.ExcelFile(input_file)
    labels = ['CORRECT', 'MISTRANSLATION', 'OMISSION', 'UNSUPPORTED_ADDITION', 'ENTITY_ERROR', 'NUMBER_OR_TIME_ERROR', 'OTHER']
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        sheets_written = 0
        
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet).iloc[:100]
            
            ref_col = next((col for col in df.columns if 'reference' in str(col).lower()), None)
            pred_col = next((col for col in df.columns if 'prediction' in str(col).lower()), None)
            
            if ref_col and pred_col:
                df = df.dropna(subset=[ref_col, pred_col])
                total_rows = len(df)
                
                if total_rows > 0:
                    df = df.rename(columns={ref_col: 'Reference', pred_col: 'Prediction'})
                    df = df.apply(evaluate_pipeline, axis=1)
                    
                    # --- CALCULATE IN-COLUMN STATISTICS ---
                    blank_row = {col: '' for col in df.columns}
                    count_row = {col: '' for col in df.columns}
                    count_row['Reference'] = 'TOTAL COUNT'
                    pct_row = {col: '' for col in df.columns}
                    pct_row['Reference'] = 'PERCENTAGE (%)'
                    
                    for label in labels:
                        count = df[label].sum()
                        pct = (count / total_rows * 100) if total_rows > 0 else 0
                        
                        count_row[label] = count
                        pct_row[label] = f"{pct:.2f}%"
                        
                    df = pd.concat([df, pd.DataFrame([blank_row, count_row, pct_row])], ignore_index=True)
                    df = df.rename(columns={'Reference': ref_col, 'Prediction': pred_col})
                    df.to_excel(writer, sheet_name=sheet, index=False)
                    
                    sheets_written += 1
                    print(f"Processed sheet: [{sheet}]")
            else:
                print(f"Skipped sheet [{sheet}]: Missing Reference/Prediction columns.")
        
        if sheets_written == 0:
            pd.DataFrame({'Message': ['No valid data found']}).to_excel(writer, sheet_name="Empty", index=False)
            
    print(f"Done. Results saved to: {output_file}")