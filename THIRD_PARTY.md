# Nguồn và thành phần kế thừa

- **SurgFormer**: https://github.com/xmed-lab/SurgFormer, commit `73c0a931f6caf8eecf82aea1ec6803a2f59b57c8`. Source archive và bản giải nén giữ trong AHA. Authors/paper credit trong upstream README. Commit tải về không có file LICENSE cấp repository được tìm thấy; không tự gán license cho source/checkpoint/dataset và không coi bundle review nội bộ là một lần public release. `aha/surgformer.py` là adapter từ mã IPL local, có PyTorch fallback và compatibility fix, không sửa source upstream.
- **SurgFormer checkpoint**: bản local đã dùng trong IPL; sha256 và thông tin epoch trong `data/manifests/assets.json` và extraction spec. Có checkpoint không có nghĩa AHA đã train lại backbone.
- **Encoder ảnh đóng băng (mặc định DINOv2 ViT-L/14)**: `facebook/dinov2-large` trên Hugging Face (model card khai báo Apache-2.0), tải bằng `scripts/fetch_encoder.py`, commit được ghim trong `assets/encoders/<name>/source.json`. Weights không nằm trong bundle.
- **MiniLM**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, pinned revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; model card khai báo Apache-2.0. Files được copy từ cache local đúng revision, không gọi mạng lúc core runtime.
- **einops 0.8.1**: https://github.com/arogozhnikov/einops, MIT; bản vendor phục vụ local review, server requirements pin cùng version.
- **Qwen3-VL-4B-Instruct**: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct, model card Apache-2.0; code comparator đã có, weights tải riêng bằng revision trong `assets/models.lock.json`.
- **TMVP / MedHorizon**: dữ liệu đã có trong workspace; provenance nhãn/frames và snapshot M1 được ghi trong manifests. Quyền sử dụng/phân phối dữ liệu tiếp tục theo nguồn, không được tái cấp phép bởi code mới.
- **PCJD/IPL snapshots**: lưu phục vụ so sánh lịch sử, không phải source runtime được yêu cầu từ thư mục cha.

Không có tuyên bố rằng các module chính AHA là reproduction được tác giả paper xác nhận. Các trích dẫn phương pháp nằm trong `proposal.md`.
