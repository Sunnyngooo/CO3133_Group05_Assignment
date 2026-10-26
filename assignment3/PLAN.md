# Kế hoạch Assignment 3: phân loại tin tức đa phương thức

Ngày lập: 14/10/2026. Trạng thái: người dùng đã chọn N24News; bộ tải ảnh, EDA/split scripts và bản proposal đã được tạo; chưa ghi nhận giảng viên duyệt và chưa triển khai thí nghiệm chính.

## 1. Phạm vi và quyết định thiết kế

- Bài toán: ảnh và tiêu đề của cùng một bài báo -> một nhãn chủ đề.
- Dataset đã chọn: N24News. Metadata tải ngày 14/10/2026 có 61.218 records và 24 chủ đề; xem [DATASET.md](DATASET.md). Full-data preparation đã xác minh đủ 61.218 cặp ảnh/headline, không có exclusions. Đây là kiểm tra kỹ thuật, chưa phải kiểm chứng ngữ nghĩa toàn bộ ảnh.
- Câu hỏi chính: fusion có tận dụng ảnh để cải thiện phân loại khi tiêu đề thiếu thông tin không?
- Câu hỏi thứ hai: tương tác giữa token văn bản và vùng ảnh qua cross-attention có lợi hơn ghép các vector đặc trưng đã pooling không?
- Metric chính: macro-F1 trên đủ các lớp của phạm vi được duyệt. Metric bổ sung: accuracy, F1 từng lớp, confusion matrix và chi phí tính toán.
- Môi trường dự kiến: Windows, RTX 4060 Laptop khoảng 8 GB VRAM đã kiểm tra trong phiên làm việc trước. Kiểm tra lại môi trường và CUDA khi bắt đầu triển khai.
- Phạm vi ban đầu: encoder ảnh ResNet-18 pretrained và encoder text DistilBERT pretrained, đều đóng băng; huấn luyện projection, fusion và classifier. Đây là lựa chọn triển khai của kế hoạch dựa trên CNN và encoder-only Transformer đã học.
- Fusion cải tiến chính là cross-attention một tầng. Gated fusion đã đề xuất trước đó có thể là phương án dự phòng nếu triển khai attention không ổn định; phải sửa proposal và thiết kế ablation tương ứng nếu đổi hướng.
- Full fine-tuning, CLIP, contrastive loss và auxiliary losses là mở rộng; ưu tiên hoàn thành bốn cấu hình và một ablation trước.

Scope chính theo yêu cầu người dùng: toàn bộ 61.218 metadata records, không cap mỗi lớp;
chỉ loại các cặp thiếu/hỏng theo audit. Giữ class imbalance gốc, seed split 44.
Run chính `full_seed44_original`; subset 12.000 mẫu chỉ giữ để phát triển code.
Đã chuẩn bị đủ 61.218 cặp; train 48.976, validation 6.122, test 6.120.
Đã kiểm tra grouping, manifest hashes và file integrity; review ngữ nghĩa bởi nhóm còn chờ.
Pipeline và lệnh thực tế ở [README.md](README.md), proposal ở [PROPOSAL.md](PROPOSAL.md).
Ảnh cho scope chính lấy cùng public asset path, bỏ query-only rendering parameters
sau khi kiểm tra ảnh mẫu có banner Published-year. Đây là image source variant được
ghi lại, không giả định giống từng pixel với ZIP của tác giả.

N24News còn hai việc phải xác minh: quyền sử dụng dataset và khả năng tải đầy đủ cặp ảnh-text. License của bài báo không tự động là license của dataset. README của tác giả chưa nêu rõ license dataset. Nếu chưa giải quyết được, ghi trạng thái này trong proposal và xin hướng dẫn giảng viên; không tự ghi một license suy đoán. Chỉ liên hệ tác giả nếu người dùng/nhóm chủ động thực hiện hoặc cho phép gửi liên hệ.

## 2. Kiến thức bài giảng được áp dụng

Các số trang dưới đây là số trang PDF, tính cả trang đầu.

| Nội dung học | Áp dụng trong Assignment 3 | Tài liệu trong thư mục cha |
|---|---|---|
| Dataset, preprocessing, leakage | Manifest, EDA, split theo nhóm, chỉ fit preprocessing trên train | `02-foundation/02-Datasets.pdf`, trang 15, 24 |
| Loss và classification metrics | Cross-entropy, accuracy, macro-F1, F1 từng lớp | `02-foundation/05-Losses.pdf`; `02-foundation/06-Metrics.pdf`, trang 19 |
| CNN/ResNet/pretrained models | Encoder ảnh, residual connections, lấy feature map trước global pooling | `03-cnn/08-CNN-Architectures.pdf`, trang 17-18 |
| Encoder-only Transformer | Encoder tiêu đề, tokenization, attention mask, pooling | `05-transformers/03-Model-Families-BERT-GPT-T5-ViT.pdf`, trang 5-6 |
| Head-only training và hiệu quả bộ nhớ | Đóng băng backbone, chọn checkpoint bằng validation, AMP, accumulation nếu cần | `05-transformers/04-Practical-Finetuning-and-Efficiency.pdf`, trang 20, 25-26, 41-43 |
| Fusion, cross-attention | Baseline concatenation và cải tiến tương tác text-image | `11-mm/mm.ENE.pdf`, trang 14, 16-21, 52-58 |
| Multi-head và combined loss | Mở rộng auxiliary supervision nếu có thời gian; không gọi nhiều head cùng dự đoán chủ đề là nhiều task khác nhau | `11-mm/mm.ENE.pdf`, trang 9-10, 86-90 |

Assignment 3 không yêu cầu triển khai tất cả mô hình đã học. Mô hình được chọn phải phục vụ câu hỏi đa phương thức và đủ các so sánh bắt buộc.

## 3. Mốc nộp chính thức và lịch nội bộ

Theo handbook trong repo, mục 7.6; mọi mốc chính thức là 23:59 GMT+7, trừ khi LMS thông báo thay đổi.

| Mốc | Hạn chính thức | Nội dung tối thiểu |
|---|---|---|
| M1 Proposal, 15% | 18/11/2026 | Dataset, modalities, pair validity, split unit, metrics, baselines; lưu trạng thái duyệt |
| M2 Draft, 25% | 25/11/2026 | Unimodal A, unimodal B, simple fusion có kết quả định lượng |
| M3 Final, 60% | 02/12/2026 | Improved fusion, ablation và toàn bộ sản phẩm nộp |

Lịch đề xuất dưới đây là mục tiêu nội bộ, không thay đổi hạn chính thức. Nên gửi proposal sớm để có thời gian chỉnh sửa và huấn luyện sau duyệt.

| Thời gian dự kiến | Công việc | Điều kiện hoàn thành |
|---|---|---|
| 14-18/10 | Kiểm tra nguồn/điều kiện sử dụng, bản tải mẫu, schema, EDA sơ bộ | Có số liệu thật về mẫu hợp lệ, nhãn, chất lượng cặp; ghi vấn đề chưa giải quyết |
| 19-23/10 | Chốt cleaning, split/subset, kiến trúc và ước lượng tài nguyên; hoàn thiện proposal | Proposal có đầy đủ trường của mục 27, sẵn sàng gửi giảng viên |
| Sau khi được Approved hoặc Approved with conditions | Thực hiện điều kiện duyệt, cài môi trường, data loader, kiểm tra batch, cache features | Pipeline đúng, đo được VRAM và thời gian; manifest/split cố định |
| Cuối tháng 10 đến 08/11, nếu đã được duyệt | Train ba baseline, kiểm tra metric, log/checkpoint | U1/U2/F1 trong bảng mô hình có kết quả trên validation và mã tái lập |
| 09-17/11, nếu đã được duyệt | Cross-attention, ablation, lặp seeds cho cấu hình chính | Đủ so sánh có kiểm soát, kết quả gắn với cấu hình và checkpoint |
| 18-24/11 | Khóa thiết kế, chuẩn bị bản Draft; đánh giá test theo protocol đã khóa | Ba baseline có kết quả định lượng và nội dung Draft đầy đủ |
| 25/11-01/12 | Hoàn thiện analysis, report, slides, video, website, reproduction | Tất cả link và hướng dẫn chạy được kiểm tra |
| 02/12 | Đại diện nhóm nộp PDF trên LMS | Tên file `G-05_A3_Report.pdf` và có đủ links trên landing page |

Nếu duyệt muộn, dời các bước huấn luyện theo thời điểm được duyệt, bỏ các mở rộng và ưu tiên ba baseline cho M2, improved fusion + ablation cho M3. Không làm thí nghiệm chính trước duyệt để giữ lịch nội bộ.

## 4. Giai đoạn trước duyệt: dữ liệu, EDA và proposal

### Bước 1: kiểm tra nguồn và bản tải

- Dùng repository/bản dữ liệu tác giả công bố; ghi URL, ngày tải, revision/hash và điều kiện sử dụng xác minh được.
- Đọc mẫu trước để xác định schema thực tế; manifest dự kiến: `sample_id`, `article_id`, `image_id`, `image_path`, `headline`, `label`, `group_id`, `split`.
- Giữ nguyên dữ liệu gốc; mọi bước cleaning có log số mẫu và lý do bị loại.
- Không ghép ảnh từ dataset khác; không dùng tên lớp làm văn bản đầu vào.
- Nhãn, URL chứa tên category, tên thư mục và metadata tiết lộ nhãn không được truyền vào encoder.

### Bước 2: EDA và cleaning

- Đếm tổng record, cặp hợp lệ và số mẫu mỗi lớp trước/sau cleaning.
- Tỷ lệ ảnh thiếu/hỏng, tiêu đề thiếu/rỗng, nhãn lỗi, ID lặp; độ dài tiêu đề và tỷ lệ truncation theo tokenizer dự kiến.
- Kiểm tra resolution/aspect ratio, exact image hash và near duplicates; tiêu đề/bài trùng hoặc gần trùng.
- Xem tối thiểu khoảng 5 cặp mỗi lớp nếu dữ liệu cho phép, ghi ví dụ đúng, nhiễu, ít liên quan và có thể mâu thuẫn. Đây là mục tiêu nội bộ, không phải ngưỡng handbook.
- Quy tắc ban đầu: loại toàn bộ sample không đủ hai phương thức khỏi bảng so sánh chính, báo cáo số lượng; cả bốn mô hình dùng cùng tập mẫu còn lại.
- Không loại mẫu vì mô hình đoán sai hoặc vì cặp khó; giữ các cặp hợp lệ nhưng khó để phân tích.
- Đầu ra: notebook EDA, bảng thống kê, biểu đồ, các ví dụ và cleaning log.

### Bước 3: split và subset

- Nếu bản tải có split tác giả cung cấp, kiểm tra duplicate/entity overlap trước khi quyết định giữ. Bài báo mô tả chia 8:1:1; chưa khẳng định bản tải hiện tại chứa sẵn split.
- Mọi record của cùng bài/ảnh trùng/nhóm sự kiện xác định được phải ở cùng split. Không mặc định tên người là group ID vì một người có thể xuất hiện trong nhiều sự kiện độc lập.
- Nếu phải tự chia: group-aware train/validation/test gần 80/10/10, cố gắng giữ phân bố lớp, seed split 44; lưu danh sách ID và quy tắc group.
- Trong preparation đã triển khai, normalized exact headlines cũng được group bảo thủ để không có text trùng giữa splits. Quy tắc này kiểm soát memorization, không coi headline trùng là bằng chứng cùng event.
- Không thể nhận diện đầy đủ event-level overlap thì mô tả rõ quy tắc khả thi và hạn chế còn lại; không tuyên bố hết leakage chỉ từ random split.
- Dùng toàn bộ cặp hợp lệ theo yêu cầu người dùng; không tự giảm về subset khi gặp giới hạn tài nguyên. Điều chỉnh batch/cache/chunk và thời gian chạy sau benchmark. Cả bốn cấu hình và ablation dùng cùng manifest full-data.
- Chọn phạm vi/subset trước khi xem kết quả mô hình; lưu ID để không đổi tập đánh giá giữa các lần chạy. Nếu giảm chỉ train thì giữ validation/test cố định và ghi rõ.
- Sau cleaning/subsetting, kiểm tra lại ngưỡng classification của handbook: ít nhất 5 lớp và khoảng 5.000 cặp; exceptions cần được duyệt.

### Bước 4: proposal

- Problem, input/output, đơn vị dự đoán, động lực thực tế và khó khăn.
- Dataset name, source, version, license/điều kiện đang xác minh, annotations và số cặp thực dùng.
- EDA sơ bộ và chất lượng ghép cặp; cleaning, split unit, cách ngăn leakage, subset rules.
- Bốn cấu hình, loss, metric, giả thuyết và ablation.
- GPU, ngân sách chạy, cách đo/ước lượng và fallback khi VRAM không đủ.
- Lưu bản proposal, feedback, trạng thái Approved/Approved with conditions/Rejected và ngày duyệt. Điều kiện duyệt phải được phản ánh trong cấu hình thí nghiệm.

## 5. Kiến trúc và bốn cấu hình bắt buộc

Pipeline chung: paired data -> preprocessing -> Dataset/DataLoader -> encoders -> representation -> fusion/head -> loss -> optimization -> prediction -> evaluation.

### Encoder và biểu diễn chung

- Ảnh: resize/crop theo pretrained weights về 224 x 224; normalization theo weights. ResNet-18 lấy feature map trước global pooling, dự kiến `(B, 512, 7, 7)` -> `(B, 49, 512)`; xác nhận shape bằng batch thật.
- Text: tokenizer của `distilbert/distilbert-base-uncased`, dùng headline, max length khởi đầu 64 tokens; output dự kiến `(B, T, 768)`, có attention mask. Nếu đổi checkpoint phải đọc config thực tế và sửa dimensions.
- Projection của hai modality về `D=256`. Giữ chung kiểu projection/pooling trong các so sánh fusion; weights được train riêng cho mỗi run.
- Image pooling: mean trên 49 vị trí. Text pooling: masked mean, bỏ padding.
- Encoder đóng băng ở eval mode, không cập nhật BatchNorm hay dropout của encoder trong pha cache. Cache theo sample ID, checkpoint revision, preprocessing, tokenizer/max length, dtype và split.
- Cache cả spatial image features và text token features để cross-attention còn đầu vào theo vị trí; chỉ lưu vector đã pooling là không đủ.

| ID | Cấu hình | Head/fusion | Loss |
|---|---|---|---|
| U1 | Image-only | Project -> image mean pool -> MLP -> 24 logits | `CE(logits_img, y)` |
| U2 | Text-only | Project -> masked mean pool -> MLP -> 24 logits | `CE(logits_txt, y)` |
| F1 | Simple fusion | Concat pooled image/text -> MLP -> 24 logits | `CE(logits_fused, y)` |
| F2 | Improved fusion | Text-to-image cross-attention -> residual -> pooling -> concat image -> MLP -> 24 logits | `CE(logits_fused, y)` |

F1 và F2 là fusion ở intermediate representation level. F2 dùng `Q=text`, `K=V=image`, một tầng multi-head attention, khởi đầu 4 heads. Query padding không được tính khi pooling hay tính loss phân tích; valid image positions làm K/V. Không suy luận đây là learned aligned CLIP space: hai encoder độc lập được kết nối qua projection/fusion trainable.

Cross-attention dự kiến: `Z_txt = LayerNorm(H_txt + MHA(H_txt, H_img, H_img))`; pooling Z_txt có text mask. Dùng cùng normalization/pooling và cấu trúc classifier khi so sánh remove-attention. Attention visualization chỉ là minh họa, không tự chứng minh grounding chính xác vì dataset không có region-level ground truth.

Baseline đơn phương thức có loss riêng khi train riêng. Phiên bản fusion chính dùng CE cho output fused, không cộng các loss phụ chưa được định nghĩa. Nếu bổ sung auxiliary image/text heads: `L_total = L_fused + lambda_img * L_img + lambda_txt * L_txt`; phải thêm so sánh có/không auxiliary losses, chọn lambda bằng validation, theo dõi từng head và gọi đúng là auxiliary supervision cho cùng bài toán chủ đề.

## 6. Giai đoạn sau duyệt: triển khai, huấn luyện và kiểm tra

1. Tạo môi trường riêng, cài PyTorch/torchvision CUDA tương thích driver, Transformers và thư viện phân tích; pin phiên bản sau khi chạy được. Không nâng driver theo suy đoán; kiểm tra tương thích từ tài liệu chính thức ở thời điểm cài.
2. Dataset/DataLoader trả ảnh, text/mask, label và ID đúng hàng. Kiểm tra một batch, shape, label range, mask/padding và thứ tự cache.
3. Chạy thử backward và overfit khoảng 32-64 sample sau duyệt để phát hiện lỗi trước các run chính. Kiểm tra macro-F1/accuracy bằng tập dự đoán nhỏ đã biết đáp án; không chỉ nhìn loss giảm.
4. Đo feature extraction, head training, peak VRAM và disk cache trên 100-500 mẫu. Ước lượng theo số sample/step thực tế; chưa ghi một thời gian train cụ thể khi chưa đo.
5. Cache features ra disk theo chunks, đọc từng batch; không đưa toàn bộ cache vào GPU. Với tối đa 64 tokens và 49 image positions, float16 raw features có quy mô khoảng 9,1 GB cho 61.218 mẫu, chưa tính metadata/overhead; số thật phụ thuộc số cặp hợp lệ, padding và định dạng cache.
6. Train U1, U2, F1 trước; kiểm tra curves, metric, checkpoint, sample predictions rồi mới F2.
7. Lưu config, split hash, model revision, seed, log, checkpoint và Git commit/tag của mỗi run. Evaluation không augment ngẫu nhiên.

Cấu hình khởi đầu để khảo sát, không phải hyperparameter đã chứng minh tối ưu:

- Image size 224; headline max length 64; fusion dimension 256, 4 attention heads.
- AdamW, head learning rate khảo sát `{1e-3, 3e-4}`, dropout 0,1; tối đa 20 epochs, early stopping patience 3 theo validation macro-F1.
- Batch size khởi đầu 32, giảm xuống 16/8 khi benchmark cần; AMP nếu phù hợp, accumulation để giữ effective batch khi cần. Ghi batch thật của mỗi run.
- Seed split 44 cố định; training seeds 44, 45, 46 cho các cấu hình chính nếu thời gian cho phép. Đây là mục tiêu nội bộ, không phải yêu cầu bắt buộc ba seeds.
- Chọn checkpoint theo validation macro-F1, dùng validation loss làm tie-break đã quy định trước. Hyperparameter search có ngân sách bằng nhau; không chọn theo test.
- Khi chuyển sang fine-tuning, cache cũ không còn hợp lệ; chạy encoder online và benchmark lại. Không trộn frozen baseline với fine-tuned fusion rồi quy toàn bộ chênh lệch cho fusion.

## 7. Thiết kế thí nghiệm và ablation

| Thí nghiệm | Giả thuyết/yếu tố thay đổi | Giữ cố định | Đầu ra và tiêu chí |
|---|---|---|---|
| E1: U1/U2/F1 | Kết hợp hai modality giúp hơn chỉ một modality | Cùng samples/split, frozen encoder checkpoints, preprocessing, tiêu chí chọn checkpoint, ngân sách tuning | Accuracy, macro-F1, per-class F1; so F1 với unimodal tốt hơn, ghi cả trường hợp giảm |
| E2: F1/F2 | Token interaction có lợi hơn concat global features | Cùng encoder/token cache, split, projection dimensions, pooling/head tương thích, ngân sách training | Chênh lệch macro-F1, F1 theo lớp, thời gian và số tham số; không bảo đảm attention sẽ thắng |
| E3: F2/remove-attention | Cải tiến có phụ thuộc vào cross-attention không? | Giữ projection, residual normalization, pooling, classifier và recipe | Train lại phiên bản dùng `Z_txt = LayerNorm(H_txt)`; đo cùng metrics, giải thích chênh lệch và thay đổi số tham số |

E3 là ablation bắt buộc, dự kiến một run cấu hình bổ sung. Có thể tái dùng F1 nếu thiết kế F1 trùng chính xác với phiên bản remove-attention và ghi rõ; không gọi hai cấu hình khác pipeline là cùng ablation. Kết quả tốt hơn phải được đọc cùng độ biến động qua seeds và chi phí, không chỉ lấy run cao nhất. Nếu chỉ có một seed thì nêu hạn chế; không tuyên bố ý nghĩa thống kê.

Chốt architecture/hyperparameters/checkpoint selection bằng train/validation trước khi chạy test. Test được dùng cho bảng kết quả cuối và phân tích định tính sau khi khóa thiết kế, không dùng để chọn mô hình.

Mở rộng chỉ khi phần bắt buộc đã xong: late fusion từ U1/U2 với trọng số chọn trên validation; headline vs abstract phải train lại các cấu hình liên quan và giữ split; auxiliary supervision; hoặc fine-tune một phần encoder với baseline đối chứng tương ứng.

## 8. Phân tích kết quả và chi phí

- Một bảng chung: config/run ID, accuracy, macro-F1, per-class metrics, total/trainable parameters, train time và peak VRAM.
- Báo cáo feature extraction riêng với head training; inference end-to-end phải tính cả encoders, tokenizer/preprocessing và fusion, không chỉ tốc độ đọc cache. Ghi batch size, warmup, GPU synchronization và số lần đo.
- Curves train/validation; confusion matrix; phân tích lớp ít mẫu, chủ đề gần nhau, title ngắn/mơ hồ và ảnh chỉ có thông tin chung chung.
- Lưu predictions theo ID. Nhóm ví dụ: text sai nhưng fusion đúng; image sai nhưng fusion đúng; unimodal đúng nhưng fusion sai; cả ba sai. Báo số lượng/tỷ lệ và kiểm tra thủ công trước khi gán nguyên nhân support/conflict.
- Nếu muốn kết luận mô hình thực sự dùng ảnh, bổ sung phân tích validation với modality ablation; chỉ suy từ prediction groups là chưa đủ bằng chứng nhân quả.
- Missing modality: bảng chính dùng cặp đầy đủ; inference kiểm tra đầu vào và trả lỗi rõ ràng khi thiếu. Fallback sang unimodal hoặc robustness study chỉ khi đã thiết kế, đánh giá và ghi riêng.
- Viết limitations: dữ liệu một nguồn, event overlap chưa loại hết nếu có, frozen encoder/domain shift, subset nếu dùng, số seed và tài nguyên.

## 9. Tổ chức mã và sản phẩm dự kiến

Các tên dưới đây là kế hoạch triển khai, chưa phải lệnh/file đã tồn tại. Dùng `assignment3/` để nhất quán với thư mục `assignment1/` hiện có; sửa README khi mã thực sự được tạo vì README hiện dùng tên `assignment_3/` chưa tồn tại.

```text
assignment3/
  PLAN.md
  README.md
  requirements.txt
  notebooks/eda.ipynb
  prepare_data.py
  extract_features.py
  dataset.py
  models.py
  train.py
  evaluate.py
  analyze_errors.py
configs/assignment3.yaml
data/assignment3/          # local only: raw, processed, split manifests, cache
checkpoints/assignment3/   # checkpoint hoặc hướng dẫn reconstruct
outputs/assignment3/       # run logs, metrics, predictions, figures
docs/assignment3.md        # trang kết quả công khai
```

Khi triển khai, thêm ignore cho dữ liệu, cache, venv, outputs nặng và checkpoint; version-control split IDs/config/cleaning rules phù hợp quyền sử dụng. Cập nhật config hiện đang `device: cpu` sau khi CUDA được xác minh. Cập nhật hướng dẫn Windows và thông tin hardware thực tế; không giữ tuyên bố Linux CPU nếu không đúng run của Assignment 3.

Đầu ra cuối:

- Proposal và approval record.
- Report ba phần: Problem & Data; Methodology; Implementation Results, có limitations/conclusion và references.
- Source, config, dependencies, seeds, train/evaluate commands, log-to-checkpoint-to-commit mapping và checkpoint hoặc hướng dẫn reconstruct.
- Assignment page và links trên landing page GitHub Pages.
- Slides và video YouTube Public/Unlisted, tiêu đề `CO3133-Semester-261 - Group G-05 - Assignment 3`; mọi thành viên trình bày/defend phần mình hiểu, có support/conflict cases.
- AI disclosure ở landing page, assignment page, report và `AI_USAGE.md`: tool/model, người dùng, thời điểm, mục đích, phần ảnh hưởng, prompt mẫu/log, cách kiểm chứng và người chịu trách nhiệm. Bao gồm việc dùng AI để thiết kế kế hoạch và tìm dataset nếu nhóm dùng nội dung này.
- Đại diện nhóm nộp `G-05_A3_Report.pdf` lên LMS; tên nhóm đăng ký chính thức là G-05.

## 10. Checklist đối chiếu handbook

- [ ] Task và dataset được duyệt; thực hiện điều kiện duyệt nếu có.
- [ ] Ít nhất hai phương thức liên quan đúng; paired sample count và ngưỡng được kiểm tra sau cleaning.
- [ ] EDA có pair counts, missing rates, text length, pair quality, label distribution.
- [ ] Split theo đúng unit; lưu IDs/seed, không entity/group overlap.
- [ ] U1 image-only, U2 text-only, F1 simple fusion có kết quả định lượng cho Draft.
- [ ] F2 improved fusion và ít nhất một ablation hoàn thành cho Final.
- [ ] Encoder, representation, alignment, fusion stage/operation, per-modality/overall losses, training và pretraining/fine-tuning strategy được mô tả.
- [ ] Mỗi thí nghiệm có hypothesis, changed/fixed factors, metrics, conclusion criteria.
- [ ] Accuracy, macro-F1, qualitative examples, support/conflict, error analysis và compute cost.
- [ ] Hardware/software, seeds, checkpoint selection và traceability được ghi.
- [ ] Report, slides, YouTube, website, source, checkpoint/reconstruction, AI disclosures và LMS PDF đủ.

## 11. Nguồn tham khảo

- Yêu cầu: [Handbook v2](../handbook-ene-v2.pdf), mục 2-5, 7.6 và 24-32; trang PDF 9-12, 17, 28-33.
- Các bài giảng sau là tài liệu local trong thư mục cha, không được đóng gói trong Git repo:
  - Multimodal: `11-mm/mm.ENE.pdf`.
  - Fine-tuning và efficiency: `05-transformers/04-Practical-Finetuning-and-Efficiency.pdf`.
  - Transformer model families: `05-transformers/03-Model-Families-BERT-GPT-T5-ViT.pdf`.
  - CNN architectures: `03-cnn/08-CNN-Architectures.pdf`.
  - Datasets, Losses, Metrics: `02-foundation/02-Datasets.pdf`, `02-foundation/05-Losses.pdf`, `02-foundation/06-Metrics.pdf`.
- [N24News paper](https://aclanthology.org/2022.lrec-1.729.pdf), [repository tác giả](https://github.com/billywzh717/N24News), kiểm tra nguồn ngày 14/10/2026.
- [ResNet-18 official documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).

Nguồn dataset/model được dùng để thiết kế, không thay thế số liệu thực nghiệm và kiểm tra thực tế. Các thiết lập, thời gian và kiến trúc chưa triển khai ở đây đều là dự kiến.
