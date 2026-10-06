# Reflection — Lab 19

**Tên:** Nguyễn Văn An  
**Cohort:** A20-K4 (VinUniversity AICB Program)  
**Path đã chạy:** lite  

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trên golden set:
- **`exact` queries:** BM25 vượt trội (96.7% vs 88.7% của semantic) do query chứa đúng thuật ngữ kỹ thuật nguyên văn (Kubernetes, JWT, OAuth) có sẵn trong inverted index. Hybrid giữ vững 96.7%.
- **`paraphrase` queries:** Do corpus tiếng Việt và model bge-small-en-v1.5 được train trên tiếng Anh, semantic thuần giảm điểm (24.0% vs BM25 33.3%). Hybrid đạt 32.0%, duy trì độ bền vững.
- **`mixed` queries:** Hybrid thắng tuyệt đối (100.0% so với BM25 97.0% và Semantic 98.5%). RRF kết hợp tín hiệu từ vựng chính xác lẫn ngữ nghĩa bối cảnh, bổ trợ lẫn nhau hoàn hảo.

**Khi không dùng hybrid:**
1. *Pure BM25:* Khi tìm kiếm mã lỗi, SKU sản phẩm, ID log, API endpoints hoặc khi thiết bị biên cực yếu không đủ RAM/CPU chạy embedding model.
2. *Pure Vector:* Khi tìm kiếm đa phương tiện (ảnh/audio), tìm kiếm đa ngữ xuyên ngôn ngữ (cross-lingual không chung từ khóa), hoặc khi câu hỏi mang tính khái niệm mở hoàn toàn không có biệt ngữ cố định.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là hiện tượng "Recall Cliff" ở NB5 khi post-filter sập về 0.00 recall lúc độ chọn lọc giảm xuống 3.8%, và mức "lift ảo" +0.120 AUC do rò rỉ dữ liệu khi dùng latest-join thay vì PIT join ở NB8.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _(Làm độc lập)_
