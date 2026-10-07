# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

`attn_only` và `correct` **hoà** nhau trên target (cả hai 0.9375) dù `attn_only`
chỉ đặt adapter trên `q, v` và nâng rank lên 283 để bù ngân sách tham số. Tôi
vào lab với kỳ vọng `text-linear @ r=16` sẽ thắng `attn @ r=283` (vì gắn vào
nhiều layer hơn → linh hoạt hơn). Thực tế với n=8 target eval, hai cấu hình
không tách được. Có thể n=8 quá nhỏ, hoặc có thể ở task JSON-triage đơn giản
này Qwen3.5-4B không cần đến tất cả các lớp MLP — `q, v` đã đủ. Đây là bằng
chứng thực nghiệm ủng hộ *LoRA Without Regret* (Thinking Machines 2025) hơn là
"all-linear luôn thắng" mà deck §11.2 gợi ý.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Mất nhiều thời gian nhất ở việc **đợi Colab chạy NB4 (45–60 phút cho 3 run)**.
Không phải chỗ dự đoán — tôi dự đoán NB3 train mới lâu nhất vì nó full data,
thực tế NB4 mới là bottleneck vì nó chạy **3 lần liên tiếp** (attn_only +
wrong_lr + qlora) cùng batch size. Bài học: khi thiết kế sweep, phải tính
tổng thời gian = baseline + N × contrast, không phải baseline + 1 contrast.

Cũng mất ~5 phút debug lúc đầu vì quên set `MASK_MODE=assistant-only` (mặc
định ban đầu là `everything`) — mask proof fail với `supervised_fraction=0.94`,
phải quay lại NB1 set lại env var. Đáng lẽ đọc `.env.example` trước.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin rằng **train loss là chỉ báo tốt để chọn adapter**. Lab cho thấy:
- `attn_only` có train loss thấp nhất (0.5377) nhưng target hoà với `correct` (0.6252, 0.9375).
- `qlora` train loss 0.7058 nhưng target 0.8438 — *cao hơn* `correct` mặc dù loss cao hơn.
- `wrong_lr` train loss 1.5702, target 0.000 — cả hai đều xấu, nhưng target
  "đúng" hơn loss vì nó đo thứ ta thật sự quan tâm.

→ Đòn bẩy chọn adapter là **điểm đo proxy đúng** (target trên task đánh giá),
không phải loss nhiếp. Bất kỳ khi nào loss và metric thật đi ngược nhau, hãy
tin metric thật.

Cũng không còn tin rằng "rank cao = tốt hơn" mà không có ngân sách so sánh.
`attn_only @ r=283` dù rank gấp 17× `correct @ r=16` vẫn hoà vì ngân sách tham
số khớp — rank chỉ có nghĩa khi so sánh **cùng vị trí**.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Cursor agent để:
- Đọc kết quả từ Colab (file JSON, CSV) và tổng hợp thành bảng trong REPORT.md.
- Viết phần diễn giải (rubric 4.2, 4.4 yêu cầu ≥150 từ kết luận + phản tư).
- Gỡ rối layout cây thư mục khi kết quả unzip ra `content/Day21-Track3-Finetuning-Lab/`
  thay vì thẳng vào lab root.

Chỗ AI sai / tôi phải sửa tay:
- Lúc đầu nó gợi ý "chỉnh `OPTIMIZED_PROMPT` để (b) thắng" — **sai liêm chính**,
  đây là cách gian lận verify gate. Tôi giữ prompt nguyên bản và ghi SHA vào report.
- Nó đề xuất xoá `valid_trace_rate=0.0` khỏi report vì "nghe xấu" — sai,
  đây là kết quả đúng cho task yêu cầu "chỉ trả JSON", tôi giữ lại và giải thích.
- Nó từng gợi ý commit `adapter_model.safetensors` lên git "cho đủ" — sai vì
  mỗi file 130 MB, repo phình 500 MB; tôi thêm `*.safetensors` vào `.gitignore`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

**Đóng băng eval set và đo 3 baseline trước khi đụng vào train loop.**

Cụ thể:
1. Lấy 50–200 mẫu vàng từ khách hàng, ghi SHA256 của từng file `eval_*.jsonl`
   vào một file manifest. **Không bao giờ sửa file eval sau khi thấy kết quả.**
2. Đo baseline (a) base + naive prompt, (b) base + optimized prompt (do con
   người viết, không phải tự sinh), (c) zero-shot với few-shot prompt khác.
   Freeze SHA của mỗi prompt.
3. Tính Δ = score(c) − score(b). Nếu Δ ≥ +0.20 mới đáng train.
4. Nếu train: dùng LoRA r=16, text-linear, **LR × 10** so với full-FT
   (1e-4 cho Qwen 4B), batch hiệu dụng < 32, 30 step đầu tiên là sanity check
   trên 8 mẫu trước khi chạy full.
5. Sau train, eval trên cùng tập vàng + regression set. Nếu regression tụt
   > 5%, dừng lại — đó là dấu hiệu overfit vào domain khách hàng, không phải
   "đã học được task".

→ Lab này dạy tôi: **pipeline đúng quan trọng hơn metric đẹp**. Một run
target=0.7 nhưng có baseline + regression + mask-proof xanh đáng tin hơn run
target=0.95 không có baseline.