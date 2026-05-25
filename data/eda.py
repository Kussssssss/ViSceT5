"""
data/eda.py
EDA tools: VQA_EDA_Analyzer, TWA_UpperBound_Analyzer, Threshold_Sensitivity_Analyzer.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm
import editdistance
from concurrent.futures import ThreadPoolExecutor
import warnings
import re

EDA = False

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

class VQA_EDA_Analyzer:
    def __init__(self, df, name="Dataset", img_root_col='image_path', ocr_root_col='ocr_path'):
        self.df = df
        self.name = name
        self.img_col = img_root_col
        self.ocr_col = ocr_root_col
        print(f"Dataset: {name} | Samples: {len(df)}")
    
    def _load_ocr_data(self, path):
        try:
            if not os.path.exists(path): return None
            if path.endswith('.npy'):
                data = np.load(path, allow_pickle=True)
                if data.ndim == 0: 
                    return data.item() 
                return data.tolist()
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return None

    def _normalize_text(self, text):
        if not isinstance(text, str): return str(text)
        text = text.lower().strip()
        if text.endswith('.'):
            text = text[:-1].strip()
        return text

    def _get_samples(self, sample_n):
        if sample_n is None or sample_n >= len(self.df):
            return self.df
        return self.df.sample(n=sample_n, random_state=42)

    def _measure_image(self, path):
        try:
            with Image.open(path) as img: return img.size
        except: return None

    def analyze_images(self, sample_n=None):
        print("ANALYZING IMAGES...")
        samples = self._get_samples(sample_n)
        img_paths = samples[self.img_col].astype(str).tolist()
        
        widths, heights, ratios = [], [], []
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(self._measure_image, img_paths), total=len(img_paths)))
            
        for res in results:
            if res:
                w, h = res
                widths.append(w); heights.append(h); ratios.append(w/h)
        
        if not widths: return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(widths, kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title("Image Width")
        
        sns.histplot(heights, kde=True, ax=axes[1], color='orange')
        axes[1].set_title("Image Height")
        
        sns.histplot(ratios, kde=True, ax=axes[2], color='green')
        axes[2].set_title("Aspect Ratio")
        axes[2].axvline(1.0, color='red', linestyle='--')
        plt.tight_layout()
        plt.show()
        
        print(f"Avg Ratio: {np.mean(ratios):.2f}")

    def analyze_text_lengths(self):
        print("ANALYZING TEXT LENGTHS...")
        q_lens = self.df['question'].fillna("").astype(str).apply(lambda x: len(x.split()))
        a_lens = self.df['answer'].fillna("").astype(str).apply(lambda x: len(x.split()))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.histplot(q_lens, bins=30, ax=axes[0], color='purple')
        axes[0].set_title("Question Length (Tokens)")
        
        sns.histplot(a_lens, bins=30, ax=axes[1], color='teal')
        axes[1].set_title("Answer Length (Tokens)")
        plt.tight_layout()
        plt.show()

    def analyze_ocr_quality(self, sample_n=None):
        print("ANALYZING OCR QUALITY...")
        samples = self._get_samples(sample_n)
        
        box_counts = []
        clean_box_counts = []
        confidences = []
        CONF_THRESHOLD = 0.5
        
        for _, row in tqdm(samples.iterrows(), total=len(samples)):
            path = str(row.get(self.ocr_col))
            data = self._load_ocr_data(path)
            
            if not data: continue
            
            items = []
            if isinstance(data, dict):
                if 'boxes' in data and 'texts' in data:
                    sc = data.get('scores', [1.0] * len(data['boxes']))
                    if isinstance(sc, np.ndarray): sc = sc.tolist()
                    items = [{'conf': s} for s in sc]
                elif 'data' in data and isinstance(data['data'], list):
                    items = data['data']
            elif isinstance(data, list):
                items = data
            
            if not items and isinstance(data, np.ndarray):
                items = data.tolist()

            box_counts.append(len(items))
            
            valid_items = 0
            for item in items:
                conf = 0.0
                if isinstance(item, dict):
                    conf = float(item.get('conf', item.get('score', item.get('confidence', 0))))
                elif isinstance(item, list) and len(item) > 1:
                     conf = 1.0 
                
                if conf > 0: confidences.append(conf)
                if conf >= CONF_THRESHOLD: valid_items += 1
            
            clean_box_counts.append(valid_items)

        if not confidences: return

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.histplot(confidences, bins=20, ax=axes[0], color='green', kde=True)
        axes[0].set_title("Confidence Distribution")
        axes[0].axvline(CONF_THRESHOLD, color='red', linestyle='--')
        
        sns.histplot(box_counts, color='gray', alpha=0.3, ax=axes[1], label='Raw Boxes')
        sns.histplot(clean_box_counts, color='blue', ax=axes[1], label=f'Clean Boxes (>{CONF_THRESHOLD})')
        axes[1].set_title("Boxes per Image")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        
        print(f"Avg Raw Boxes: {np.mean(box_counts):.1f}")
        print(f"Avg Clean Boxes: {np.mean(clean_box_counts):.1f}")

    def analyze_intersection(self, sample_n=None):
        print("ANALYZING INTERSECTION (RECALL)...")
        samples = self._get_samples(sample_n)
        stats = {'exact': 0, 'fuzzy': 0, 'total': 0}
        
        for _, row in tqdm(samples.iterrows(), total=len(samples)):
            ans_raw = str(row['answer'])
            ans = self._normalize_text(ans_raw)
            
            if not ans or ans == 'nan': continue
            
            data = self._load_ocr_data(str(row.get(self.ocr_col)))
            if not data: continue
            
            ocr_texts = []
            if isinstance(data, dict) and 'texts' in data:
                raw_ocr = data['texts']
                if isinstance(raw_ocr, np.ndarray): raw_ocr = raw_ocr.tolist()
                ocr_texts = [self._normalize_text(t) for t in raw_ocr]
            elif isinstance(data, list):
                for x in data:
                    if isinstance(x, dict):
                        t = x.get('text', x.get('transcription', ''))
                        ocr_texts.append(self._normalize_text(t))
            
            if not ocr_texts: continue
            stats['total'] += 1
            
            match_found = False
            
            if ans in ocr_texts:
                match_found = True
            
            if not match_found:
                if any(ans in t for t in ocr_texts):
                    match_found = True
            
            if not match_found:
                for t in ocr_texts:
                    if not t: continue
                    if editdistance.eval(ans, t) / max(len(ans), 1) < 0.3:
                        match_found = True; break
            
            if match_found:
                stats['exact'] += 1
        
        if stats['total'] == 0: return
        print(f"Recall (Normalized): {stats['exact']/stats['total']:.1%}")


if EDA:
    analyzer = VQA_EDA_Analyzer(final_train_df, name="OpenViVQA", img_root_col='image_path', ocr_root_col='ocr_path')
    analyzer.analyze_images()
    analyzer.analyze_text_lengths()
    analyzer.analyze_ocr_quality()
    analyzer.analyze_intersection()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import unicodedata
import re
import warnings

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class TWA_UpperBound_Analyzer:
    def __init__(self, df, img_root_col='image_path', ocr_root_col='ocr_path'):
        self.df = df
        self.img_col = img_root_col
        self.ocr_col = ocr_root_col
        print(f"🚀 Initialized TWA Upper Bound Check on {len(df)} samples...")

    def _normalize_token(self, text):
        if not isinstance(text, str): return str(text)
        text = unicodedata.normalize('NFC', text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _load_ocr_tokens_set(self, path):
        try:
            if not os.path.exists(path): return set()
            
            data = None
            if path.endswith('.npy'):
                temp = np.load(path, allow_pickle=True)
                data = temp.item() if temp.ndim == 0 else temp.tolist()
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            if not data: return set()

            raw_texts = []
            if isinstance(data, dict) and 'texts' in data: 
                raw_texts = data['texts']
                if hasattr(raw_texts, 'tolist'): raw_texts = raw_texts.tolist()
            elif isinstance(data, list):
                for x in data:
                    if isinstance(x, dict): 
                        raw_texts.append(x.get('text', x.get('transcription', '')))
            
            token_set = set()
            for t in raw_texts:
                norm_t = self._normalize_token(str(t))
                token_set.update(norm_t.split()) 
            
            return token_set
        except:
            return set()

    def classify_question(self, question):
        q = self._normalize_token(str(question))
        text_keywords = [
            'chữ', 'số', 'tên', 'biển', 'viết', 'hiệu', 'tiêu đề', 
            'nội dung', 'dòng', 'thương hiệu', 'địa chỉ', 'sđt', 'điện thoại',
            'ngày', 'tháng', 'năm', 'bao nhiêu', 'mã'
        ]
        if any(k in q.split() for k in text_keywords):
            return 'Text-VQA Questions'
        return 'Visual-VQA Questions'

    def calculate_strict_ub(self):
        results = []
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Checking Strict UB"):
            ans_raw = str(row['answer'])
            ans_norm = self._normalize_token(ans_raw)
            ans_tokens = ans_norm.split()
            
            if not ans_tokens: 
                results.append({'group': 'All', 'strict_match': False})
                continue

            ocr_tokens_set = self._load_ocr_tokens_set(str(row.get(self.ocr_col)))
            
       
            is_strict_match = all(token in ocr_tokens_set for token in ans_tokens)
            
            q_type = self.classify_question(row['question'])
            
            results.append({'group': 'Whole Dataset', 'strict_match': is_strict_match})
            results.append({'group': q_type, 'strict_match': is_strict_match})

        df_res = pd.DataFrame(results)
        
        summary = df_res.groupby('group')['strict_match'].mean() * 100
        summary = summary.reset_index().sort_values('strict_match', ascending=False)
        
        print("\n📊 --- STRICT UPPER BOUND ACCURACY (No char error) ---")
        print(summary)
        
        plt.figure(figsize=(8, 6))
        
        plot_data = summary[summary['group'].isin(['Text-VQA Questions', 'Visual-VQA Questions', 'Whole Dataset'])]
        
        bars = plt.bar(plot_data['group'], plot_data['strict_match'], color='#4472C4', width=0.5)
        
        plt.title('Strict Upper Bound Accuracy (SwinTextSpotter)', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.legend([bars], ["Strict UB - No character error allowed."], loc='upper right')
        
        plt.show()

if EDA:
    analyzer = TWA_UpperBound_Analyzer(final_train_df, img_root_col='image_path', ocr_root_col='ocr_path')
    analyzer.calculate_strict_ub()

import pandas as pd
import os
import json
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import re
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def normalize_and_tokenize(text):
    if not isinstance(text, (str, int, float)): return []
    text = str(text)
    text = unicodedata.normalize('NFC', text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

def load_ocr_file(path):
    if not path or not os.path.exists(path): return None
    try:
        if path.endswith('.npy'):
            data = np.load(path, allow_pickle=True)
            if data.ndim == 0: return data.item()
            return data.tolist()
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return None
    return None

def extract_ocr_with_scores(data):
  
    items = []
    
    if data is None: return []

    if isinstance(data, dict):
        texts = []
        scores = []
        
        for k in ['texts', 'transcriptions', 'words', 'ocr_tokens']:
            if k in data and isinstance(data[k], list):
                texts = data[k]
                break
        
        for k in ['scores', 'confidences', 'probs']:
            if k in data and isinstance(data[k], list):
                scores = data[k]
                break
        
        if texts:
            for i, txt in enumerate(texts):
                sc = scores[i] if (scores and i < len(scores)) else 1.0
                items.append((txt, float(sc)))
            return items

        if 'data' in data and isinstance(data['data'], list):
             data = data['data'] # Chuyển xuống Case 2 xử lý

    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict):
                txt = x.get('text', x.get('transcription', x.get('words', '')))
                sc = x.get('score', x.get('conf', x.get('confidence', x.get('prob', 1.0))))
                if txt:
                    items.append((txt, float(sc)))
            elif isinstance(x, str):
                items.append((x, 1.0))

    return items

class Threshold_Sensitivity_Analyzer:
    def __init__(self, df, dataset_name="Dataset", img_col='image_path', ocr_col='ocr_path'):
        self.df = df
        self.name = dataset_name
        self.ocr_col = ocr_col
        print(f"🚀 Initialized Sensitivity Check for {dataset_name} ({len(df)} samples)")

    def run_analysis(self, thresholds=np.arange(0.0, 1.0, 0.05)):
        strict_hits = {t: 0 for t in thresholds}
        total_samples = 0
        
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Scanning Dataset"):
            # 1. Prepare Answer
            raw_ans = row.get('answer', '')
            ans_tokens = normalize_and_tokenize(raw_ans)
            if not ans_tokens: continue
            
            # 2. Prepare OCR
            ocr_path = str(row.get(self.ocr_col))
            ocr_data = load_ocr_file(ocr_path)
            raw_items = extract_ocr_with_scores(ocr_data)
            
            processed_items = []
            for txt, score in raw_items:
                tokens = normalize_and_tokenize(txt)
                if tokens:
                    processed_items.append((tokens, score))
            
            total_samples += 1
            
            # 3. Test các ngưỡng Threshold trong RAM
            for t in thresholds:
                valid_ocr_tokens = set()
                
                for tokens, score in processed_items:
                    if score >= t:
                        valid_ocr_tokens.update(tokens)
                
                is_match = True
                for token in ans_tokens:
                    if token not in valid_ocr_tokens:
                        is_match = False
                        break
                
                if is_match:
                    strict_hits[t] += 1
        
        if total_samples == 0: return

        accuracies = [(strict_hits[t] / total_samples * 100) for t in thresholds]
        
        best_acc = max(accuracies)
        best_thresh = thresholds[accuracies.index(best_acc)]
        
        print(f"\n🏆 BEST THRESHOLD: {best_thresh:.2f} (Recall: {best_acc:.2f}%)")
        print(f"📉 DROP at 0.90: {accuracies[list(thresholds).index(0.90)]:.2f}%")

        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, accuracies, marker='o', linestyle='-', linewidth=2, color='#2b5797')
        
        plt.axvline(best_thresh, color='red', linestyle='--', label=f'Best: {best_thresh:.2f}')
        
        plt.title(f'Impact of OCR Confidence Threshold on Upper Bound Recall\n({self.name})', fontsize=14)
        plt.xlabel('Confidence Threshold (Min Score to keep)', fontsize=12)
        plt.ylabel('Strict Upper Bound Accuracy (%)', fontsize=12)
        plt.xticks(np.arange(0, 1.01, 0.1))
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        for i, acc in enumerate(accuracies):
            if i % 2 == 0: 
                plt.text(thresholds[i], acc + 0.5, f"{acc:.1f}", ha='center', fontsize=9)
        
        plt.show()

if EDA:
    analyzer = Threshold_Sensitivity_Analyzer(final_train_df, dataset_name="OpenViVQA Train")
    analyzer.run_analysis()