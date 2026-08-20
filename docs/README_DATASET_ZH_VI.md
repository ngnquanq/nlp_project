# README_DATASET  
# Historical Hán–Việt Parallel Corpus (Chinese ↔ Vietnamese)

---

# 1. Giới thiệu dataset

**Tên dataset:**

`Historical Hán–Việt Parallel Corpus`

**Mục đích sử dụng:**

Dataset phục vụ:

- Nhóm 09:
  - **Chinese-to-Vietnamese MT for Ancient Texts**

- Nhóm 10:
  - **Vietnamese-to-Chinese MT for Ancient Texts**

Đây là corpus dịch máy trong miền:

- Hán văn;
- văn bản lịch sử;
- văn bản cổ;
- ngôn ngữ Hán–Việt.

---

# 2. Cấu trúc dataset

```
HNVI_Historical_MT/

├── train/
│
├── dev/
│
├── test/
│
└── README_DATASET.md
```

---

# 3. Thống kê dataset

## Training split

Khoảng:

```
19.000+ cặp song ngữ
```

Cụ thể:

```
19,218 parallel pairs
```

---

## Development split

```
510 parallel pairs
```

---

## Test split

```
510 parallel pairs
```

---

# 4. Hướng dịch

Dataset gốc:

```
Hán cổ
   ↓
Tiếng Việt
```

---

## Nhóm 09

Sử dụng:

```
Chinese → Vietnamese
```

---

## Nhóm 10

Sử dụng cùng dataset nhưng đảo chiều:

```
Vietnamese → Chinese
```

---

# 5. Đặc điểm ngôn ngữ

Dataset chứa các hiện tượng khó của dịch máy lịch sử.

---

## 5.1. Thuật ngữ lịch sử

Ví dụ:

- chức quan;
- triều đại;
- thể chế;
- thuật ngữ hành chính;
- cụm Hán–Việt cố định.

---

## 5.2. Thực thể lịch sử

Bao gồm:

- nhân vật;
- vua/chúa;
- quan chức;
- địa danh;
- tổ chức;
- triều đại.

---

## 5.3. Thông tin thời gian

Bao gồm:

- năm;
- niên hiệu;
- thời kỳ lịch sử;
- thứ tự sự kiện.

---

## 5.4. Đặc điểm Hán văn

Văn bản Hán cổ thường có:

- câu ngắn;
- nhiều hàm ý;
- thiếu chủ ngữ;
- phụ thuộc ngữ cảnh;
- từ đa nghĩa.

---

# 6. Quy định dữ liệu

Train/dev/test split đã được chuẩn bị.

Sinh viên:

## Được phép

- sử dụng dataset cho đồ án;
- preprocessing nếu cần;
- xây dựng mô hình MT.

## Không được phép

- thay đổi test split;
- sử dụng test data để training;
- public dataset;
- upload dataset công khai.

---

# 7. Experiment

## Bắt buộc

### Experiment 1

Fairseq baseline

### Experiment 2

LLM adaptation

Khuyến nghị:

```
Qwen3-8B-Instruct
+
QLoRA
```

---

# 8. Bonus Experiment

## knowledge_v2

Dành cho Nhóm 09 và Nhóm 10.

Mục tiêu:

So sánh:

```
Fairseq baseline

vs

Fairseq + knowledge_v2
```

Code và configuration được cung cấp.

Sinh viên không cần tự thiết kế knowledge injection architecture.

---

# 9. Giá trị nghiên cứu

Dataset phục vụ:

- dịch máy Hán–Việt;
- nghiên cứu low-resource MT;
- knowledge-enhanced MT;
- phân tích lỗi dịch;
- nghiên cứu hallucination trong MT.

Các lỗi quan trọng cần chú ý:

- sai thực thể;
- sai chức danh;
- sai niên hiệu;
- sai số liệu;
- thêm thông tin không có trong nguồn.
