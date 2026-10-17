# Hướng dẫn chạy demo (Representation Engineering)

Thư mục này chứa **mã nguồn gốc** của bài báo *"Representation Engineering: A Top-Down Approach to AI Transparency"*
(Zou et al., arXiv:2310.01405), lấy từ repo chính thức của tác giả:
https://github.com/andyzoujm/representation-engineering

> **Lưu ý:** Theo yêu cầu đề bài, thư mục này **chỉ chứa source code, không chứa dữ liệu** (`data/` đã bị loại bỏ khi đóng gói `source.zip`).
> Notebook demo bên dưới tự tải dữ liệu cần thiết trực tiếp từ repo công khai khi chạy.

## Demo: Honesty Detection & Control (Kaggle Notebook)

**File:** `examples/honesty/honesty_kaggle_demo.ipynb`

Bản chuyển thể (adapt) từ notebook gốc `honesty.ipynb` của tác giả, chạy trên **Kaggle Notebooks**
với mô hình **`NousResearch/Llama-2-7b-chat-hf`** — một bản mirror công khai, **không bị Meta gate**,
cùng trọng số/cấu hình hệt `meta-llama/Llama-2-7b-chat-hf` — nạp ở dạng **4-bit** (vừa GPU T4/P100
miễn phí của Kaggle).

> **Vì sao dùng mirror thay vì `meta-llama/Llama-2-7b-chat-hf` trực tiếp?**
> Model gốc của Meta bị "gate" — phải bấm "Agree and access repository" rồi **chờ Meta duyệt**,
> và việc duyệt **không phải lúc nào cũng tức thì** (có thể mất vài giờ đến vài ngày, đôi khi báo lỗi
> `403 ... awaiting a review`). Dùng mirror `NousResearch/Llama-2-7b-chat-hf` để không phải chờ —
> vẫn đúng là model LLaMA-2-7B-Chat, không cần xin quyền.

### Cách chạy

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → New Notebook → File → Import Notebook → chọn
   `examples/honesty/honesty_kaggle_demo.ipynb`.
2. **Bật GPU**: Settings → Accelerator → **GPU T4 x2** (hoặc P100).
3. **Bật Internet**: Settings → Internet → **On** (bắt buộc, Kaggle tắt mặc định — nếu quên, `pip install` và tải model sẽ lỗi).
4. Chạy từng cell từ trên xuống. Với model mirror mặc định, **không cần** Kaggle Secrets / HF token.

Notebook sẽ:
- Trích xuất "honesty direction" bằng phương pháp **LAT** (Section 3.1 của bài báo).
- Vẽ biểu đồ độ chính xác phân loại honest/untruthful theo từng layer.
- Dựng một **lie detector** theo từng token trên một tình huống có động cơ nói dối.
- Áp dụng **Representation Control** để ép mô hình trả lời trung thực hơn, so sánh output có/không can thiệp.

### Các lỗi đã biết & cách xử lý (đã sửa sẵn trong notebook)

| Lỗi | Nguyên nhân | Đã xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'repe'` | `pip install git+...` thất bại âm thầm (thường do Internet đang tắt) | Notebook tự dò, nếu import lỗi sẽ tự `git clone` + thêm vào `sys.path` làm fallback |
| `AttributeError: 'RepReadingPipeline' object has no attribute 'framework'` | Thư viện `repe` (viết từ 2023) giả định API nội bộ cũ của `transformers`; bản `transformers` mới đã đổi cách set `self.framework` | Notebook tự set `rep_reading_pipeline.framework = "pt"` ngay sau khi tạo pipeline |
| `403 ... gated repo ... awaiting a review` | Dùng `meta-llama/Llama-2-7b-chat-hf` trực tiếp, Meta chưa duyệt quyền truy cập | Notebook mặc định dùng mirror `NousResearch/Llama-2-7b-chat-hf`, không bị gate |

## Các demo khác có trong repo gốc (không chỉnh sửa, giữ nguyên để tham khảo)

| Thư mục | Nội dung |
|---|---|
| `examples/primary_emotions/` | Điều khiển cảm xúc (happiness, sadness, anger, fear, surprise, disgust) |
| `examples/fairness/` | Giảm thiên lệch giới tính/chủng tộc trong câu trả lời |
| `examples/harmless_harmful/` | Phát hiện & tăng khả năng chống jailbreak |
| `examples/memorization/` | Giảm việc mô hình lặp lại nguyên văn dữ liệu đã học (ghi nhớ) |
| `lorra_finetune/` | Mã huấn luyện LoRRA (Low-Rank Representation Adaptation — Algorithm 1 trong bài báo) |
| `repe_eval/` | Khung đánh giá định lượng (TruthfulQA, ARC, ...) bằng rep-reading |

Các demo này dùng chung thư viện lõi trong `repe/` (hai pipeline mới: `rep-reading` và `rep-control`),
giữ nguyên bản gốc của tác giả (chưa được platform-proof/debug như notebook Kaggle ở trên).

## Cài đặt (nếu chạy local/khác Kaggle)

```bash
pip install -e .          # cài thư viện repe từ pyproject.toml trong thư mục này
pip install bitsandbytes  # nếu muốn nạp model ở 4-bit trên GPU cá nhân
```

Yêu cầu: Python ≥ 3.9, PyTorch, một GPU có ít nhất ~6GB VRAM (dùng 4-bit) hoặc ~14GB (fp16) cho LLaMA-2-7B.
