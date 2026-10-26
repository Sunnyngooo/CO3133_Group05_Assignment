# Dataset Proposal - Assignment 3

**Đề tài:** Phân loại chủ đề tin tức bằng ảnh và tiêu đề: so sánh fusion đặc trưng và cross-attention.

**Nhóm:** G-05 (tên đăng ký chính thức, đã được xác nhận).
**Thành viên:** Đàm Hoài An; Nguyễn Hữu Minh Khôi; Ngô Diễm Quyên.
**Giảng viên:** Lê Thành Sách.
**Ngày chuẩn bị:** 14/10/2026.
**Trạng thái:** bản để nhóm/giảng viên review; chưa ghi nhận approval, chưa có thí nghiệm huấn luyện chính.

## 1. Bài toán và mục tiêu

Mỗi sample là một bài báo với ảnh và headline tương ứng. Input là ảnh RGB và tiêu đề tiếng Anh; output là một nhãn trong 24 chủ đề N24News. Đơn vị dự đoán là article record, target lấy từ `section`. Nhãn và metadata URL không được đưa vào encoder.

Câu hỏi nghiên cứu:

1. Kết hợp ảnh và headline có cải thiện macro-F1 so với từng phương thức riêng trên cùng tập mẫu không?
2. Cross-attention giữa token tiêu đề và vùng ảnh có ích hơn ghép vector đã pooling không?
3. Ảnh cung cấp thông tin bổ sung ở nhóm nào, và khi nào ảnh/chủ đề biên tập tạo ra tín hiệu gây nhầm?

Khó khăn dự kiến: title ngắn/mơ hồ; ảnh chân dung không mô tả trực tiếp chủ đề; các nhãn Movies/Theater/Music, Health/Well và Food/Travel có nội dung giao nhau; ảnh dùng lại giữa các bài; dữ liệu một nguồn và class imbalance.

## 2. Dataset, nguồn, phiên bản và điều kiện sử dụng

- Tên: N24News, Wang et al., LREC 2022.
- [Paper](https://aclanthology.org/2022.lrec-1.729.pdf); [repo tác giả](https://github.com/billywzh717/N24News).
- Metadata revision `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`, tải ngày 14/10/2026.
- Metadata file: 45.402.188 bytes, SHA-256 `47145500b7161cf90914c2dfc7e4d269931e06f74958d7b89d8a6d40f82b2ede`.
- Các fields thực tế: `section`, `headline`, `abstract`, `caption`, `image_url`, `article_url`, `image_id`; không có split/body trong bản metadata này.
- Annotation: single-label topic (`section`) của bài báo; không có bounding boxes, region-word alignments hoặc pair-quality labels gốc.
- **Điều kiện sử dụng đang cần xác minh:** repo chưa nêu rõ license dataset. License paper không được coi là license dữ liệu. Đề nghị giảng viên hướng dẫn/chấp thuận điều kiện sử dụng phù hợp trước thí nghiệm chính; nhóm cần ghi kết quả xác minh. Không tự gán MIT/CC-BY cho dataset.

Ảnh được tải từ cùng public asset path của URL tác giả cung cấp, bỏ query-only rendering parameters. Pilot với URL nguyên bản cho thấy nhiều ảnh có banner NYT Published-year. Đã xem trực quan 24 ảnh gốc, một/lớp: nội dung cùng subject với các mẫu metadata đã xem và không thấy banner năm ở 24 ảnh này. Đây không phải khẳng định mọi ảnh trong corpus đều không có chữ/banner. Metadata URL nguyên bản và source variant được lưu để tái lập. Phiên bản này phải được mô tả là bản asset tải lại tại thời điểm nghiên cứu, không khẳng định pixel-identical với ZIP tác giả công bố năm 2022.

## 3. Quy mô toàn bộ dữ liệu và EDA sơ bộ

Audit toàn bộ metadata đã thực hiện:

- 61.218 records thuộc 24 lớp; các lớp từ 681 đến 3.000 mẫu.
- Không rỗng section, headline, image URL, image ID hoặc article URL; caption rỗng ở 303 record.
- 304 image URLs xuất hiện nhiều lần (405 extra records theo exact URL); 1 article URL lặp (1 extra record).
- 180 headlines lặp (1.637 extra records theo exact text); title trùng chưa đủ để kết luận cùng sự kiện.
- Headline trung vị 9 từ, p95 khoảng 13 và tối đa 22 theo whitespace count; chưa phải token counts của tokenizer.

Scope đề xuất theo yêu cầu người dùng: toàn bộ 61.218 metadata records thuộc 24 lớp, không cap mỗi lớp. Giữ tất cả cặp hợp lệ sau kiểm tra ảnh/headline; mẫu thiếu/hỏng bị loại với audit log và không thay bằng mẫu chọn theo model outcome. Group từ canonical article/image URLs trên toàn metadata, sau đó bổ sung hash/perceptual/exact-text grouping trước khi chia split. Run chính `full_seed44_original`, seed split 44; các ID và hash được lưu cố định trước khi huấn luyện.

Giữ phân bố lớp quan sát được của toàn bộ cặp hợp lệ, không undersample để cân bằng. Macro-F1 là metric chính vì lớp không cân bằng. Full-data preparation đã hoàn tất, xác minh đủ 61.218 ảnh và headline hợp lệ. Custom group-aware split và URL variant vẫn phải được mô tả/đề nghị duyệt; không so trực tiếp các số với paper như cùng split/backbone hoặc cùng pixel ảnh.

Kiểm tra pair quality ban đầu: 120 cặp, 5/lớp, được Codex xem trên contact sheets và ghi reviewer trong CSV; cả 120 ID có mặt trong main selection. Đánh giá AI sơ bộ có 81 cặp có tín hiệu chủ đề rõ, 37 cặp chỉ liên quan theo ngữ cảnh, 2 cặp chưa rõ nếu thiếu caption/body. Đây không phải annotation của tác giả hoặc xác minh độc lập bởi sinh viên, và không suy rộng thành tỷ lệ chất lượng toàn dataset. Nhóm cần review lại các cases trước submission.

Kết quả thực tế của run chính:

- **61.218 cặp** mở được, **0 ảnh thất bại**, **0 exclusions**; giữ đúng số mẫu từng lớp của metadata.
- Train **48.976**, validation **6.122**, test **6.120**; mỗi lớp giữ gần tỷ lệ 80/10/10, không ép số lượng các lớp bằng nhau.
- **58.727 groups**, lớn nhất 410 records; 78 groups chứa nhiều editorial labels (189 records), không tự động coi là annotation errors.
- **3.120** perceptual-similarity links; 1.637 extra records có headline trùng theo exact text.
- Kiểm tra disjoint IDs/groups/hashes/canonical URLs/exact normalized headlines đã pass.
- File ảnh tổng cộng **6.363.053.805 bytes**. Kiểm tra độc lập SHA-256 của toàn bộ file ảnh, manifest bytes, metadata/label mapping và liên kết near-duplicate đã pass.

Đây là xác minh kỹ thuật và grouping, không phải xác minh ngữ nghĩa của tất cả 61.218 cặp.

- [Metadata audit](../outputs/assignment3/metadata_audit.json)
- [Pilot pair review](../outputs/assignment3/pilot_seed44/pair_review.csv)
- [Pilot review summary](../outputs/assignment3/pilot_seed44/pair_review_summary.json)
- [Historical development-subset summary](../outputs/assignment3/subset_500_seed44_original/eda_summary.json)
- [Full-data EDA](../outputs/assignment3/full_seed44_original/EDA.md)
- [Full-data summary](../outputs/assignment3/full_seed44_original/eda_summary.json)
- [Independent preparation verification](../outputs/assignment3/full_seed44_original/preparation_verification.json)
- [Full-data human review form](../outputs/assignment3/full_seed44_original/pair_review.csv)

Full-data manifests và label mapping nằm trong `data/assignment3/processed/full_seed44_original/`.
Không tái sử dụng split của subset cho bảng kết quả full-data.

## 4. Cleaning, pairing và chống leakage

- Nối bằng `image_id`; file ảnh gốc nằm trong `data/assignment3/raw/images_original/`.
- Kiểm tra decode, dimensions tối thiểu, file/pixel SHA-256; loại failed images và empty headlines, ghi lý do.
- Mọi mô hình dùng cùng tập cặp đầy đủ; không để image-only dùng nhiều mẫu hơn text/fusion trong bảng chính.
- Group theo canonical image URL/article URL trên full metadata và liên kết bắc cầu; bổ sung exact decoded-pixel/file duplicates trên toàn bộ cặp hợp lệ.
- Group các headline giống nhau sau chuẩn hóa whitespace/case để tránh exact-text memorization giữa splits. Đây là quy tắc bảo thủ kiểm soát text overlap, không khẳng định các bài có cùng headline là cùng sự kiện.
- Group bảo thủ các ảnh gần giống theo dHash distance <=1, RGB thumbnail RMSE <=5 và aspect delta <=0,05, bỏ flat images khỏi phép so perceptual. Đây là nhóm similarity để chống rò rỉ, không phải nhãn sự kiện gốc.
- Train/validation/test gần 80/10/10, seed split 44, giữ nguyên group và cố gắng cân bằng từng lớp; số thật theo EDA summary. Lưu manifest IDs và hashes.
- Kiểm tra sample IDs, group IDs, canonical URLs, image hashes và normalized exact headlines không giao nhau giữa splits.
- Event-level IDs đầy đủ chưa có: broader event overlap không được khẳng định đã loại hết. Review các title/event gần nhau; mô tả hạn chế còn lại.
- Fit các thống kê học từ dữ liệu trên train; validation dùng tuning/checkpoint selection. Test chỉ dùng sau khi khóa thiết kế và protocol.

## 5. Phương pháp và loss

Preprocessing dự kiến: normalization/resize theo pretrained ResNet-18 weights; headline tokenizer của `distilbert/distilbert-base-uncased`, max length ban đầu 64, padding và attention mask. Lưu phiên bản checkpoint/tokenizer và đo tỷ lệ truncation. Raw image files được bảo toàn.

Image encoder: ResNet-18 pretrained ImageNet, feature map trước global pooling -> spatial positions. Text encoder: DistilBERT pretrained, token embeddings. Đóng băng cả hai và chạy eval mode trong pha cache, lưu image spatial features và text token features theo ID. Projection về D=256, masked text pooling; chỉ fusion/head được huấn luyện trong so sánh chính.

| Cấu hình | Biểu diễn/fusion | Loss |
|---|---|---|
| U1 image-only | Image projection + mean pooling + MLP | Cross-entropy cho output ảnh |
| U2 text-only | Text projection + masked mean pooling + MLP | Cross-entropy cho output text |
| F1 simple fusion | Concatenate pooled image/text -> MLP | Cross-entropy cho output fusion |
| F2 improved fusion | Text queries image positions qua 1 tầng cross-attention, residual/normalization, pooling và classifier | Cross-entropy cho output fusion |

Fusion diễn ra ở intermediate representation level. F2 dự kiến 4 attention heads, Q=text và K/V=image. Không gọi đây là CLIP-aligned embedding space hoặc region-level grounding đã được xác minh. Loss baseline được train riêng; không thêm per-modality auxiliary loss vào fusion chính nếu chưa có experiment tương ứng.

Training recipe khởi đầu để chọn bằng validation: AdamW, learning rate `{1e-3, 3e-4}`, dropout 0,1, tối đa 20 epochs, early stopping patience 3 theo macro-F1. Batch khởi đầu 32, benchmark rồi điều chỉnh với cùng effective batch khi có accumulation. Seed split 44; mục tiêu training seeds 44/45/46 nếu ngân sách cho phép, báo số runs thật. Chọn checkpoint bằng validation macro-F1, tie-break bằng validation loss.

## 6. Thiết kế thí nghiệm và đánh giá

| Experiment | Hypothesis và thay đổi | Giữ cố định | Decision criteria |
|---|---|---|---|
| E1: U1/U2/F1 | Fusion sử dụng thông tin bổ sung tốt hơn unimodal | Samples/split, frozen encoders, preprocessing, budget tuning và checkpoint rule | Chênh lệch macro-F1 với unimodal tốt hơn, accuracy, từng lớp và chi phí |
| E2: F1/F2 | Token interaction có ích hơn global concatenation | Encoder/cache, dimensions, split, classifier và ngân sách tương thích | Macro-F1, biến động qua runs, latency/parameters; ghi cả khi F2 giảm |
| E3: remove-attention | Đóng góp của cross-attention | F2 bỏ attention nhưng giữ projection, normalization, pooling, classifier và training recipe | So với F2 bằng cùng metric và phân tích số tham số thay đổi |

Outputs bắt buộc: accuracy và macro-F1; per-class F1/confusion matrix; loss/metric curves; qualitative examples; support/conflict cases; error analysis; ablation; limitations và compute cost. Không bắt buộc fusion thắng unimodal. Attention maps chỉ minh họa, không thay bằng chứng ground truth về grounding.

Inference cost phải đo end-to-end gồm preprocessing/tokenization, encoders và fusion, tách với cached-head inference. Báo hardware, versions, total/trainable parameters, peak VRAM, feature extraction time, training time, checkpoint/config/split hashes và run IDs/commit.

## 7. Tài nguyên và kế hoạch hoàn thành

Máy cá nhân Windows, RTX 4060 Laptop khoảng 8 GB VRAM; preparation environment dùng Python 3.11 và pinned dependencies trong `requirements-data.txt`. Chưa cài/chạy training, nên chưa có số đo training time hoặc peak training VRAM.

Ước lượng dung lượng raw feature cache float16 cho tối đa 61.218 mẫu với 49 x 512 image features và tối đa 64 x 768 text features: khoảng 9,1 GB, chưa tính masks/metadata/overhead. Đọc cache theo chunks/batches, không nạp toàn bộ lên GPU. Cần benchmark 100-500 mẫu sau approval để ước lượng thời gian GPU và chốt batch; không đưa một ETA train chưa đo vào report. Budget dự kiến 4 cấu hình + 1 ablation trên cùng full-data manifests, ưu tiên complete core trước optional fine-tuning/auxiliary losses.

Các mốc chính thức theo handbook: Proposal 18/11/2026, Draft 25/11/2026 (ba baseline có quantitative results), Final 02/12/2026 (improved fusion, ablation và sản phẩm đầy đủ), 23:59 GMT+7. Nhóm gửi proposal sớm; main experiments chỉ bắt đầu khi Approved/Approved with conditions và thực hiện các điều kiện duyệt.

## 8. Những mục nhóm cần xác nhận và approval record

- [x] Group Name theo đăng ký: G-05.
- [ ] Người phụ trách Assignment 3, người nộp proposal và người chịu trách nhiệm review.
- [ ] Điều kiện sử dụng dữ liệu được xác minh/giảng viên hướng dẫn.
- [ ] Chấp thuận phạm vi toàn bộ cặp hợp lệ, original-asset URL variant và custom group-aware split.
- [ ] Review lại EDA/pair review AI; bổ sung nhận xét của thành viên.
- [ ] Lưu feedback, ngày duyệt và trạng thái Approved/Approved with conditions/Rejected.

Approval status: **not recorded**. Approval date/conditions: **chưa có**.

Sản phẩm cuối: code/config/checkpoints hoặc reconstruction, report PDF ba phần, slides, YouTube có mọi thành viên, assignment page/landing links, AI disclosures ở đủ nơi; đại diện nhóm nộp `G-05_A3_Report.pdf` trên LMS.

## 9. References

AI disclosure for this preparation: Codex assisted with dataset research,
experimental design, data-preparation code, EDA and proposal drafting. Agent
verification covers the actual full metadata/images/manifests and nine passing tests;
semantic inspection is limited to 120 pilot pairs and 24 original-asset images,
with reviewer recorded. Exact model identifier, member using the tool and final
human verifier need to be completed by the group. Detailed log and representative
prompts: [AI_USAGE.md](../AI_USAGE.md). Human review is pending; no model results
or instructor approval are claimed.

1. Z. Wang, X. Shan, X. Zhang, J. Yang. N24News: A New Dataset for Multimodal News Classification. LREC 2022, pp. 6768-6775. [Paper](https://aclanthology.org/2022.lrec-1.729/).
2. [N24News authors' repository](https://github.com/billywzh717/N24News), revision stated above.
3. [Course handbook v2](../handbook-ene-v2.pdf), mục 7.6, 27-31 và các yêu cầu chung.
4. Multimodal lecture, `11-mm/mm.ENE.pdf`, fusion/cross-attention; tài liệu học trong thư mục cha, không nằm trong Git repo.
5. Practical fine-tuning lecture, `05-transformers/04-Practical-Finetuning-and-Efficiency.pdf`; tài liệu học trong thư mục cha, không nằm trong Git repo.
6. [ResNet-18 official documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html); [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased).
