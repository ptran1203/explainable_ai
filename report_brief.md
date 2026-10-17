# Report Brief — tài liệu nguồn để viết `Report.docx`

> File này **không phải** báo cáo — nó là bản tổng hợp mọi context cần để viết `Report.docx`,
> để khi bắt tay viết (bằng tay hoặc nhờ AI), không cần lật lại toàn bộ lịch sử chat hay
> đọc lại cả PDF 56 trang. Mọi phần nội dung chi tiết (số liệu, bảng, công thức, references)
> đã có sẵn trong [`representation_engineering_notes.md`](representation_engineering_notes.md)
> — file này chỉ **trỏ tới** đúng mục cần dùng cho từng phần của report, cộng thêm phần
> Demo + Glossary chưa có ở đâu khác.

---

## 0. Bối cảnh đề bài (từ `README.md`)

- Môn: (điền tên môn — đề chỉ ghi "seminar", không có tên môn cụ thể trong README)
- Nhóm 2 người, MSHV: **25C11016** và **25C11003**
- Yêu cầu: đọc hiểu + **dịch lại bài báo theo văn phong người Việt Nam** (không phải dịch word-by-word), cài đặt/dùng source public để demo, làm slide thuyết trình seminar.
- Nộp bài: file `25C11016_25C11003.zip`, giải nén ra folder cùng tên chứa:
  - `Report.docx` ← **file này hỗ trợ viết file đó**
  - `Slides.pptx`
  - `source.zip` ← đã xong, xem [`repe_source/HUONG_DAN_DEMO.md`](repe_source/HUONG_DAN_DEMO.md)
- Script đóng gói cuối: `package.py` / `package.bat` ở gốc project.

---

## 1. Bài báo đang xử lý

**Representation Engineering: A Top-Down Approach to AI Transparency**
Zou, Phan, Chen, Campbell, Guo, Ren, et al. — Center for AI Safety / CMU / Berkeley / Stanford — arXiv:2310.01405v4
Code: https://github.com/andyzoujm/representation-engineering

Toàn bộ nội dung bài báo (đã xử lý từ PDF, giữ cấu trúc gốc) nằm trong
[`representation_engineering_notes.md`](representation_engineering_notes.md) — dùng file đó làm
**nguồn sự thật duy nhất** cho số liệu/bảng/công thức khi viết report, **không** quay lại đọc PDF gốc.

---

## 2. Cấu trúc đề xuất cho `Report.docx` + map sang nguồn nội dung

| # | Mục trong Report (tiếng Việt) | Lấy nội dung từ |
|---|---|---|
| — | Trang bìa | MSHV, tên bài báo, tên môn |
| — | Mục lục | (Word tự sinh qua Heading styles) |
| — | Tóm tắt (Abstract dịch + viết lại) | notes.md → mục "Abstract" |
| 1 | Giới thiệu & động lực | notes.md → mục "1. Introduction" + §2 Glossary bên dưới (Sherringtonian/Hopfieldian) |
| 2 | Công trình liên quan | notes.md → mục "2. Related Work" (2.1–2.3) |
| 3 | Phương pháp Representation Engineering | notes.md → mục "3. Representation Engineering" (LAT, 3 toán tử control, Algorithm 1 LoRRA) — **xem thêm phần lý thuyết mở rộng ở §3 dưới đây**, giải thích tại sao RepE "đọc được bên trong" model |
| 4 | Nghiên cứu điển hình: Honesty | notes.md → mục "4. In Depth Example of RepE: Honesty" (Table 1, 2) |
| 5 | Đạo đức & Quyền lực | notes.md → mục "5. In Depth Example of RepE: Ethics and Power" (Table 3) |
| 6 | Các hướng mở rộng (emotion, harmlessness, bias, knowledge editing, memorization) | notes.md → mục "6. Example Frontiers..." (Table 4–7) |
| 7 | Demo thực nghiệm (nhóm tự làm) | **§4 dưới đây** — chưa có sẵn ở notes.md, viết mới từ phần này |
| 8 | Kết luận & hạn chế | notes.md → mục "7. Conclusion" + "Key Takeaways" + Appendix E (X-risk, rút gọn) |
| — | Tài liệu tham khảo | notes.md → mục "References" (có thể chọn lọc ~30-40 trích dẫn quan trọng nhất thay vì cả ~140, tuỳ độ dài report được yêu cầu) |

**Lưu ý văn phong (theo đúng yêu cầu đề bài):** viết lại bằng lời của nhóm, giọng văn báo cáo học thuật
tiếng Việt — **không dịch word-by-word** từ notes.md (bản notes cũng đã là diễn giải lại từ bài báo,
không phải bản dịch nguyên văn, nên khi viết report cần diễn đạt lại một lần nữa bằng văn phong của nhóm).

---

## 3. Phần lý thuyết bổ sung (dùng cho mục 1 & 3 của report)

Toàn bộ nội dung lý thuyết "Sherringtonian vs. Hopfieldian" + giải thích RepE có "đọc được bên trong
model" hay không, đã viết sẵn (tiếng Việt, dạng slide-ready) trong
[`presentation_notes.md`](presentation_notes.md). Khi viết mục 1 (Giới thiệu) và mục 3 (Phương pháp),
kéo nội dung từ đó vào, diễn giải thành văn xuôi báo cáo (hiện đang ở dạng bullet cho slide).

Ý chính cần giữ lại khi viết văn xuôi cho report:
- Hai cách nhìn trong khoa học thần kinh nhận thức: Sherringtonian (neuron/circuit, bottom-up) vs.
  Hopfieldian (population-level representation, top-down) — RepE theo hướng Hopfieldian.
- RepE không phải "chỉ là prompt engineering": nó đọc/viết trực tiếp lên **hidden state tensor** thật
  trong forward pass (qua forward hook), không chỉ thay đổi văn bản đầu vào.
- 4 bước LAT (Linear Artificial Tomography): Collect → Model (PCA trên vector hiệu) → Read (dot product)
  → Control (cộng vector ngược vào activation).

---

## 4. Phần Demo thực nghiệm — nội dung cần viết mới cho mục 7 của report

**Chưa có ở notes.md nào cả — đây là phần duy nhất nhóm cần tự viết từ đầu (không phải diễn giải lại
bài báo), vì đây là việc nhóm tự làm.**

### Những gì đã thực hiện
- Clone repo chính thức của tác giả: `github.com/andyzoujm/representation-engineering`.
- Chạy lại thí nghiệm **Honesty** (Section 4 của bài báo) trên **Kaggle Notebooks** (GPU T4 x2 miễn phí),
  dùng notebook đã chỉnh sửa: `repe_source/examples/honesty/honesty_kaggle_demo.ipynb`.
- Model dùng: `NousResearch/Llama-2-7b-chat-hf` (mirror công khai, không bị Meta gate, cùng trọng số với
  `meta-llama/Llama-2-7b-chat-hf`), nạp ở dạng **4-bit NF4 quantization** (`BitsAndBytesConfig`) để vừa
  VRAM GPU miễn phí.

### Các bước của pipeline demo (viết thành quy trình trong report)
1. Trích xuất **honesty reading vector** bằng LAT: dataset `facts_true_false.csv` (câu đúng/sai), đặt
   trong khung prompt "Pretend you're an honest/untruthful person...", lấy hidden state ở mỗi layer,
   PCA trên vector hiệu (honest − untruthful).
2. Đánh giá độ chính xác phân loại honest/untruthful theo từng layer (tái hiện Fig. 5 / Table 8 của bài báo).
3. Dựng **lie detector** theo từng token trên tình huống có động cơ nói dối ("giết người, không muốn bị
   bắt") — tái hiện Figure 8/9.
4. **Representation Control**: cộng vector honesty (hệ số `coeff=8.0`) vào activations tại layer giữa,
   so sánh output có/không can thiệp — tái hiện Figure 10 / Table 2.

### Vấn đề gặp phải khi triển khai & cách xử lý (nên đưa vào report như một phần "bài học kinh nghiệm")

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| `403 ... gated repo ... awaiting a review` khi nạp `meta-llama/Llama-2-7b-chat-hf` | Model bị Meta "gate", yêu cầu quyền truy cập không được duyệt tức thì | Chuyển sang dùng mirror công khai `NousResearch/Llama-2-7b-chat-hf` — cùng trọng số, không bị gate |
| `ModuleNotFoundError: No module named 'repe'` | `pip install git+...` thất bại âm thầm (do Internet bị tắt mặc định trên Kaggle) | Bật Internet trong Settings; notebook cũng tự dò và `git clone` + thêm vào `sys.path` làm fallback |
| `AttributeError: 'RepReadingPipeline' object has no attribute 'framework'` | Thư viện `repe` viết năm 2023 cho API nội bộ cũ của `transformers`; bản `transformers` mới nhất đã đổi cách gán `self.framework` | Gán tay `rep_reading_pipeline.framework = "pt"` ngay sau khi tạo pipeline |

→ Đây là 3 ví dụ cụ thể, nên đưa vào report để minh chứng nhóm **thực sự chạy demo** (không chỉ đọc
code), và cũng là nội dung hay cho phần Q&A khi thuyết trình.

### Kết quả mong đợi khi chạy (để đưa hình/số liệu vào report sau khi chạy thật)
- Biểu đồ accuracy theo layer (lưu tại `/kaggle/working/layer_accuracy.png`).
- Log output: câu trả lời "No Control" vs. "+ Honesty Control" cho 2 câu hỏi test (tổng thống Mỹ 2030,
  và tình huống giết người) — minh chứng trực quan, dễ đưa vào slide/report nhất.
- **Ghi chú:** tới thời điểm viết file này, nhóm đã sửa xong lỗi code nhưng **chưa xác nhận chạy full
  thành công end-to-end** — cần chạy lại trên Kaggle, chụp màn hình/lưu output thật trước khi viết số
  liệu cụ thể vào report (đừng bịa số liệu, dùng đúng output thật chạy ra).

---

## 5. Bảng thuật ngữ Anh–Việt (dùng để nhất quán xuyên suốt Report + Slides)

| English | Tiếng Việt đề xuất |
|---|---|
| Representation Engineering (RepE) | Kỹ thuật biểu diễn (RepE) — giữ tên gốc RepE khi viết tắt |
| representation | biểu diễn |
| representation reading | đọc biểu diễn |
| representation control | điều khiển biểu diễn |
| hidden state | trạng thái ẩn |
| activation | kích hoạt (activation) |
| reading vector | vector đọc |
| contrast vector | vector tương phản |
| control vector / controller | vector điều khiển |
| LAT (Linear Artificial Tomography) | LAT (giữ tên gốc, có thể dịch chú thích "chụp cắt lớp tuyến tính nhân tạo") |
| LoRRA | LoRRA (giữ tên gốc) |
| probing / probe | dò / đầu dò (probe) |
| mechanistic interpretability | khả năng diễn giải theo cơ chế (bottom-up) |
| top-down / bottom-up | từ trên xuống / từ dưới lên |
| honesty | tính trung thực |
| truthfulness | tính xác thực / độ chân thực |
| hallucination | ảo giác (hallucination) |
| jailbreak | vượt rào (jailbreak) |
| fine-tune | tinh chỉnh (fine-tune) |
| fairness / bias | công bằng / thiên lệch (bias) |
| memorization | ghi nhớ / học vẹt |
| knowledge editing | chỉnh sửa kiến thức |
| power-seeking | xu hướng tìm kiếm quyền lực |
| quantization (4-bit) | lượng tử hoá (4-bit) |
| gated repo / gated model | mô hình bị giới hạn quyền truy cập ("gate") |
| mirror (model mirror) | bản sao công khai (mirror) |

---

## 6. Checklist trước khi coi Report.docx là xong

- [ ] Mọi số liệu/bảng lấy đúng từ `representation_engineering_notes.md`, không bịa số.
- [ ] Phần Demo (mục 7) dùng output thật từ lần chạy Kaggle, không viết "giả định sẽ ra kết quả X".
- [ ] Văn phong tiếng Việt tự viết lại, không dịch máy nguyên văn.
- [ ] Thuật ngữ nhất quán theo bảng ở mục 5.
- [ ] Có Tài liệu tham khảo (ít nhất các nguồn chính: bài báo gốc + repo code).
- [ ] Trang bìa có đủ MSHV1/MSHV2 theo đúng format đề bài.
