# Presentation Notes — Representation Engineering (RepE)

> Working file for building `Slides.pptx`. Each `##` section below is meant to become
> one slide (or a tight cluster of 2) — kept short/bulleted on purpose, with longer
> speaker-note prose underneath for whoever presents.

---

## Slide: Lý thuyết nền — Hai cách nhìn về nhận thức (cognitive neuroscience)

**Bảng chính (đưa trực tiếp vào slide dạng bảng):**

| | Sherringtonian | Hopfieldian |
|---|---|---|
| Đơn vị phân tích | Từng neuron riêng lẻ, kết nối node-to-node | Mẫu hoạt động ở cấp **quần thể** (population-level) = **representation** |
| Tương ứng trong AI | Mechanistic Interpretability (circuits, từng neuron/attention head) | **Representation Engineering** |
| Hướng tiếp cận | Bottom-up (ghép các phần nhỏ → hành vi) | Top-down (xem toàn bộ mẫu hoạt động là đối tượng nghiên cứu) |

**Luận điểm cốt lõi (nói miệng / ghi chú thuyết trình):**
- Một khái niệm như "trung thực" hay "muốn có quyền lực" khó mà nằm gọn trong 1 neuron hay 1 circuit — nó là một **hướng (direction)** trong không gian activation nhiều chiều, trải rộng trên nhiều chiều của hidden state ở một layer.
- Vì vậy, thay vì "giải mã ngược" từng circuit (cách làm của mechanistic interpretability — bài báo cho rằng cách này khó mở rộng quy mô), RepE coi cả vector hidden-state là một thể thống nhất, và hỏi: "hướng nào trong không gian này tương ứng với khái niệm mình quan tâm?"

---

## Slide: Phương pháp này có thực sự "đọc được bên trong" mô hình không?

**Câu trả lời: Có — đúng nghĩa đen, không phải ẩn dụ.**

RepE thao tác trực tiếp trên **hidden states** (các vector activation chạy trong residual stream giữa các layer transformer), lấy ra ngay trong lúc forward pass. Đây là mức độ "bên trong" sâu hơn cả logits/văn bản đầu ra, và cũng sâu hơn attention weights hay saliency map (những thứ đó chỉ nói về *input*, không phải tính toán nội tại của model).

### 4 bước cơ chế (LAT — Linear Artificial Tomography)

1. **Thu thập (Collect)** — Chạy model với các câu prompt đối lập (ví dụ: "hãy đóng vai người trung thực" vs. "hãy đóng vai người không trung thực"), ở mỗi layer **lấy đúng tensor hidden-state thật** tại 1 vị trí token nhất định. Đây là tensor nội tại thật — chính là tensor mà model dùng để tính layer kế tiếp, không phải bản dựng lại hay xấp xỉ.
2. **Mô hình hoá (Model)** — Lấy **hiệu** giữa 2 hidden state theo từng cặp (honest − untruthful), chạy PCA trên nhiều hiệu như vậy → thành phần chính đầu tiên là **"reading vector" v**. Vector này nằm trong đúng không gian vector mà hidden-state thật của model đang sống.
3. **Đọc (Read)** — Với input mới, lấy tích vô hướng giữa hidden state thật và v. Đây là một phép đọc trực tiếp vào những gì đang diễn ra bên trong forward pass — ngay cả **trước khi** model sinh ra token đầu ra. Đây chính là cách lie detector của bài báo hoạt động: nó chấm điểm hoạt động **nội tại**, không phải từ ngôn từ xuất ra.
4. **Điều khiển (Control)** — Đây là phần vượt xa "chỉ đọc": pipeline điều khiển **cộng thêm v (nhân hệ số) ngược vào hidden state** trong lúc sinh văn bản, thông qua forward hook, tại đúng layer đã trích xuất. Đây là một can thiệp nhân quả trực tiếp vào tính toán nội tại của model — forward pass thật sự bị thay đổi giữa dòng, không chỉ là đổi prompt hay lọc hậu kiểm đầu ra.

> Trong code demo (`honesty_colab_demo.ipynb` / `honesty_kaggle_demo.ipynb`):
> `rep_control_pipeline(..., activations=activations)` — `activations[layer]` được tiêm thẳng vào residual stream tại layer đó trong lúc `model.generate()` chạy.

### Vì sao điều này quan trọng — ví dụ nên đưa vào báo cáo/slide

Section 6.2 của bài báo phát hiện điều đáng chú ý nhờ "quyền truy cập bên trong" này: ngay cả khi một jailbreak khiến LLaMA-2 sinh ra nội dung có hại thành công, **điểm số nội tại cho "mức độ gây hại" vẫn cao** — nghĩa là **nhận định bên trong** của model ("cái này có hại") không đổi, chỉ có **quyết định có hành động theo nhận định đó hay không** là thay đổi. Chỉ có thể phát hiện sự khác biệt này bằng cách đọc hidden state trực tiếp; nếu chỉ nhìn văn bản đầu ra, model "trông như" đã bị lừa, nhưng bên trong thì không.

### Tóm lại — vị trí của RepE giữa 2 cực

```
Black-box prompting          Representation Engineering          Mechanistic Interpretability
(không đụng vào nội tại)  →  (đọc/viết trực tiếp lên hidden    →  (giải mã đến từng neuron/
                              state, nhưng xem là 1 hướng/         circuit/attention head)
                              subspace thống nhất, không
                              tách nhỏ theo từng neuron)
```

RepE nằm giữa: nó đọc và viết lên đúng tensor activation nội tại (giống mechanistic interpretability), nhưng xử lý chúng như các **hướng/subspace tổng thể** thay vì cố quy về từng neuron riêng lẻ (khác mechanistic interpretability, giống cách nhìn Hopfieldian).

---

## Ghi chú thêm (candidate cho slide "Related Work" hoặc Q&A backup)

- **Analogy dễ hiểu cho người nghe không chuyên ML**: giống như đo sóng não (EEG/MEG) thay vì giải phẫu từng neuron — đó cũng là lý do phương pháp được gọi là "Linear Artificial Tomography" (LAT), chơi chữ với "tomography" (chụp cắt lớp) trong y khoa/neuroimaging.
- Có thể dùng để trả lời câu hỏi phản biện kiểu: *"Vậy phương pháp này có phải chỉ là prompt engineering không?"* → Không — prompt engineering chỉ thay đổi input text, không đụng gì đến nội tại model. RepE Control thay đổi trực tiếp tensor hidden-state giữa các layer, bất kể prompt là gì.
