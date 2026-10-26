# Dataset đã chọn: N24News

Ngày chọn và tải metadata: 14/10/2026. Phạm vi chính: ảnh + headline -> section.
Người dùng đã chọn dataset; chưa ghi nhận giảng viên duyệt proposal.

## Phạm vi chính hiện tại: toàn bộ dữ liệu

Theo yêu cầu người dùng, chọn tất cả **61.218 records**, không giới hạn 500 mẫu/lớp.
Run chính: `full_seed44_original`, đã hoàn tất **61.218 cặp hợp lệ**, không loại record nào.
Ảnh tổng cộng **6.363.053.805 bytes**; tất cả file đã được kiểm tra decode/hash.
Train **48.976**, validation **6.122**, test **6.120**; có **58.727 groups**,
nhóm lớn nhất 410 records. Kiểm tra không giao nhau về ID/group/hash/URL/headline đã pass.
Kiểm tra độc lập hash của toàn bộ file ảnh và manifest cũng đã pass.
Giữ phân bố lớp quan sát được, không cân bằng bằng cách bỏ mẫu. Split gần 80/10/10
theo nhóm, seed 44. Subset và các số liệu bên dưới được giữ làm lịch sử preparation.
Không trộn manifest subset với full data. Việc mở rộng scope không có nghĩa giảng viên đã duyệt.

- Manifest: `data/assignment3/processed/full_seed44_original/`.
- [EDA toàn bộ dữ liệu](../outputs/assignment3/full_seed44_original/EDA.md).
- [Xác minh manifest](../outputs/assignment3/full_seed44_original/preparation_verification.json).
- [Ảnh để review](../outputs/assignment3/full_seed44_original/review_gallery/).
- [Bảng ghi review](../outputs/assignment3/full_seed44_original/pair_review.csv).

## Nguồn và trạng thái

- [Repository tác giả](https://github.com/billywzh717/N24News).
- [Bài báo LREC 2022](https://aclanthology.org/2022.lrec-1.729.pdf).
- Revision nguồn: `047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106`.
- Metadata đã tải: `data/assignment3/raw/nytimes_dataset.json`, 45.402.188 bytes.
- SHA-256: `47145500b7161cf90914c2dfc7e4d269931e06f74958d7b89d8a6d40f82b2ede`.
- Đã tải và xác minh decode/hash đủ 61.218 ảnh cho scope toàn bộ dữ liệu, tổng 6.363.053.805 bytes. Ảnh scope chính ở `data/assignment3/raw/images_original/`. Subset 12.000 ảnh trước đây được giữ làm lịch sử ở phần dưới.
- Đã xem 120 cặp metadata-rendered và 24 ảnh original-asset đại diện các lớp bằng AI; nhận xét được lưu rõ reviewer, vẫn cần thành viên nhóm review. Không khẳng định toàn bộ corpus đã được kiểm chứng ngữ nghĩa.
- Script gốc `build_fully_dataset.py` và dependencies `requirements.n24news.txt` đã được tải từ cùng revision, đặt cạnh JSON trong `data/assignment3/raw/`. Script chưa được thực thi.
- README của tác giả chưa nêu rõ license dataset; ghi vấn đề này trong proposal. License của bài báo không phải bằng chứng license dữ liệu.

## Schema thực tế

| Field | Vai trò |
|---|---|
| `section` | Nhãn chủ đề, chỉ dùng làm target |
| `headline` | Văn bản đầu vào chính |
| `abstract` | Văn bản bổ sung cho thí nghiệm mở rộng |
| `caption` | Mô tả ảnh, không dùng trong cấu hình chính |
| `image_url` | URL để tải ảnh |
| `article_url` | ID/nguồn để kiểm tra bài trùng, không đưa vào encoder |
| `image_id` | ID nối metadata với file ảnh |

File metadata gốc là một danh sách record, không có field split hoặc body.
Với bài toán đã chọn không cần crawl body; headline có sẵn.

Script gốc đọc `nytimes_dataset.json` từ working directory, tải ảnh vào
`./images/<image_id>.jpg`, crawl body và ghi `nytimes_dataset_full.json`.
Nó xóa các file đang có trong `./images` trước khi bắt đầu; cần tính đến hành vi
này khi chạy lại hoặc khi thư mục đã chứa ảnh. File requirements của tác giả
được lưu với tên riêng để không nhầm với dependencies của Assignment 3.

## Kiểm tra metadata sơ bộ

Các số dưới đây tính từ file đã tải, chưa phải EDA đầy đủ hoặc số cặp ảnh hợp lệ:

- 61.218 records, 24 chủ đề. Class counts từ 681 (Well) đến 3.000; dữ liệu không cân bằng hoàn toàn.
- Không thiếu section, headline, image_url, image_id hoặc article_url. Caption thiếu ở 303 record, không ảnh hưởng trực tiếp cấu hình ảnh + headline.
- Image IDs không lặp. Có 1 URL bài báo xuất hiện lặp, tương ứng 1 record dư nếu tính theo URL.
- 304 URL ảnh xuất hiện nhiều lần, tương ứng 405 record dư theo URL ảnh. Các record dùng cùng URL ảnh cần được xét cùng group khi chia dữ liệu; vẫn phải kiểm tra hash để phát hiện ảnh trùng ở URL khác.
- 180 tiêu đề xuất hiện nhiều lần, tổng 1.637 record dư theo phép đếm chuỗi tiêu đề. Tiêu đề trùng không tự chứng minh cùng bài hoặc cùng sự kiện; cần kiểm tra trước khi nhóm/loại.
- Headline có trung vị 9 từ, p95 khoảng 13 từ, tối đa 22 từ theo whitespace count; đây không phải số token của DistilBERT.
- Không có URL ảnh rỗng không đồng nghĩa mọi ảnh tải được. Image ID khác nhau cũng không bảo đảm nội dung ảnh khác nhau.

Report chi tiết: [metadata_audit.json](../outputs/assignment3/metadata_audit.json).

Chạy lại phép kiểm tra từ root repo, không cần cài thư viện ngoài Python:

```powershell
python assignment3/audit_metadata.py --source-revision 047a2bc3f4df0ea738b847a4ac3a6bd6aa2e4106 --retrieved-date 2026-10-14
```

## Subset đã chuẩn bị trước đây (chỉ dùng phát triển code)

- Seed chọn mẫu/split 44; 500 records/lớp, không chọn theo model outcomes.
- Source variant: cùng public image asset path, bỏ query-only rendering parameters. Mẫu URL nguyên bản có banner Published-year; 24 ảnh original-asset đã xem không thấy banner này. Không giả định từng pixel giống bản ZIP của tác giả.
- Lượt tải đầu có 16 lỗi DNS tạm thời; lượt thử lại đã tải đủ, cuối cùng 0 failed images. Các ảnh thử từ metadata URLs được giữ riêng trong `raw/images/` để lưu provenance.
- Train **9.600**, validation **1.200**, test **1.200**; mỗi lớp lần lượt 400/50/50.
- **11.632 groups** sau URL/hash/perceptual/exact-text grouping; nhóm lớn nhất 62 records. Các headline trùng được group bảo thủ để kiểm soát memorization, không coi đó là event IDs.
- Kiểm tra không giao nhau giữa splits: sample IDs, groups, file/pixel hashes, canonical image/article URLs và normalized exact headlines. Broader event overlap chưa được khẳng định loại hết.
- Manifest, label mapping và download audits: `data/assignment3/processed/subset_500_seed44_original/`.
- [EDA report](../outputs/assignment3/subset_500_seed44_original/EDA.md), [EDA summary](../outputs/assignment3/subset_500_seed44_original/eda_summary.json), [Proposal](PROPOSAL.md), [lệnh chạy lại](README.md).

## Bước tiếp theo

Review EDA và điều kiện sử dụng, điền người phụ trách cho nhóm G-05, gửi proposal
giảng viên và lưu approval record. Sau duyệt mới thực hiện thí nghiệm chính.

Kế hoạch kiến trúc, thí nghiệm và sản phẩm nộp: [PLAN.md](PLAN.md).
