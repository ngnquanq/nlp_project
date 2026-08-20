# README_DATASET  
# VietEng Ancient Corpus (Vietnamese ↔ English)

## 1. Giới thiệu dataset

**Tên dataset:**

`VietEng_ancient`

**Mục đích sử dụng:**

Dataset phục vụ hai đề tài:

- Nhóm 07:
  - **English-to-Vietnamese MT for Ancient Texts**

- Nhóm 08:
  - **Vietnamese-to-English MT for Ancient Texts**

Dataset tập trung vào các văn bản văn học Việt Nam cổ điển, có đặc điểm:

- ngôn ngữ văn học;
- nhiều từ ngữ cổ;
- nhiều biểu thức văn hóa;
- cấu trúc câu khác biệt với tiếng Việt hiện đại;
- yêu cầu mô hình hiểu ngữ nghĩa thay vì chỉ dịch theo từ vựng.

---

# 2. Cấu trúc dataset

```
VietEng_ancient/

├── core/
│
├── bonus/
│
└── README_DATASET.md
```

---

# 3. Core Dataset (bắt buộc sử dụng)

## 3.1. Truyện Kiều

### File tiếng Việt

```
KieuTale1870_vie.txt
```

### File tiếng Anh

```
KieuTale_eng_2.txt
```

### Vai trò

Đây là bộ dữ liệu chính cho experiment của Nhóm 07 và Nhóm 08.

Direction:

```text
Vietnamese ↔ English
```

Trong đó:

- Nhóm 07:

```
English → Vietnamese
```

- Nhóm 08:

```
Vietnamese → English
```

---

## Quy định về bản dịch Truyện Kiều

Trong experiment chính thức:

**Chỉ sử dụng:**

```
KieuTale_eng_2.txt
```

làm bản dịch tiếng Anh chuẩn.

Các bản dịch Truyện Kiều khác không sử dụng trong main experiment.

Lý do:

- tránh hiện tượng một tác phẩm có nhiều reference translation;
- tránh leakage giữa các phiên bản;
- đảm bảo các nhóm có cùng một evaluation target.

---

# 3.2. Chinh Phụ Ngâm

## File tiếng Việt

```
CPN_vie.txt
```

## File tiếng Anh

```
CPN_eng_1.txt
```

Vai trò:

- bổ sung dữ liệu văn học cổ;
- giúp mô hình học được phong cách dịch văn học;
- tăng khả năng tổng quát hóa ngoài một tác phẩm duy nhất.

---

# 4. Bonus Dataset

## 4.1. Hồ Xuân Hương

File:

```
HXH_vie.txt
HXH_eng.txt
```

Vai trò:

- đánh giá khả năng generalization;
- kiểm tra mô hình trên tác phẩm không xuất hiện trong training.

Khuyến nghị:

Không sử dụng làm training data trong experiment chính.

---

## 4.2. Lục Vân Tiên

File:

```
LVT_vie.txt
LVT_eng.txt
```

Vai trò:

Bonus track:

- nghiên cứu alignment;
- tạo thêm parallel data;
- đánh giá trên tác phẩm khác.

Lưu ý:

Alignment giữa tiếng Việt và tiếng Anh có thể không hoàn toàn 1-1.

Có thể xuất hiện:

```
1 câu tiếng Việt ↔ nhiều câu tiếng Anh
```

hoặc:

```
nhiều câu tiếng Việt ↔ 1 câu tiếng Anh
```

Nếu sử dụng cần kiểm tra alignment trước.

---

# 5. Quy định sử dụng dữ liệu

Sinh viên phải:

- giữ nguyên cấu trúc dataset;
- ghi rõ preprocessing;
- ghi rõ dữ liệu bổ sung nếu sử dụng;
- phân biệt:
  - dữ liệu gốc;
  - dữ liệu bổ sung;
  - dữ liệu sinh tự động.

Không được:

- public dataset;
- upload dataset lên GitHub/Kaggle công khai;
- chia sẻ dataset ngoài phạm vi môn học.

---

# 6. Experiment khuyến nghị

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

# 7. Giá trị nghiên cứu

Dataset này phục vụ:

- nghiên cứu dịch máy văn học cổ;
- domain adaptation;
- phân tích lỗi dịch;
- nghiên cứu hallucination trong MT.

Các lỗi cần chú ý:

- dịch sai ý nghĩa;
- bỏ sót thông tin;
- thêm thông tin không có trong nguồn;
- sai tên riêng;
- sai sự kiện/văn hóa.
