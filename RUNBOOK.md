# Chạy AHA v2 trên máy review và server

Mọi đường dẫn tính từ folder `AHA`. Chưa có bước nào dưới mục 3 được chạy trên GPU.

## 1. Kiểm tra local (CPU)

```powershell
$env:PYTHONPATH="${PWD}/src;${PWD}/vendor"
$env:OMP_NUM_THREADS='2'
python -m pytest -q
python -m aha audit
python -m aha verify
```

`verify` đọc hash toàn bộ payload (hơn 21 GB). `audit` kiểm tra cấu trúc, cả QA gốc và QA dẫn xuất, và liệt kê cache feature đã có.

## 2. Chuyển sang server

Cần cả `data/frames`, `data/labels`, `data/private`, manifests, `assets`, `vendor`, không chỉ source. Frame ZIP và checkpoint local là hard link; không sửa tại chỗ. `scripts/package.py --output D:/transfer/AHA-v2.tar` tạo TAR nếu cần (ổ đích phải đủ chỗ, không đặt trong AHA).

Luồng chuẩn: **GitHub chỉ chứa source code** (src, configs, scripts, tests, tài liệu chạy); mọi thứ còn lại nằm trong dataset Hugging Face **private**: frame ZIP ở `part_*/`, và nhãn, QA, gold, manifests, checkpoint SurgFormer, mã SurgFormer vendored, MiniLM, references, `proposal.md`, `FREEZE.json` ở `bundle/<protocol>/`.

Trên máy có đủ folder AHA, sau mỗi lần freeze:

```bash
python scripts/upload_bundle.py --repo NhatHuy1110/AHA        # từ chối nếu dataset là public
```

Trên server:

```bash
git clone https://github.com/NhatHuy1110/AHA.git && cd AHA
python -m pip install huggingface_hub && hf auth login
python scripts/fetch_data.py --repo NhatHuy1110/AHA --protocol AHA-M1-v2.0
```

Sau khi cài môi trường (mục 3), `python -m aha verify` phải khớp toàn bộ hash. Dữ liệu TMVP/MedHorizon vẫn theo điều khoản gốc: không chuyển dataset sang public khi chưa kiểm tra quyền phân phối lại.

## 3. Môi trường server

Linux, Python 3.10–3.12, một GPU từ 24 GB. Chưa có phép đo VRAM thực tế của v2.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.txt
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD/src:$PWD/vendor"
python -m pytest -q
```

## 4. Encoder và cache feature

Encoder mặc định là DINOv3 ViT-L/16. Model này bị gate trên Hub: chấp nhận licence trên trang model và `hf auth login` trước.

```bash
python scripts/fetch_encoder.py --id facebook/dinov3-vitl16-pretrain-lvd1689m --name dinov3_vitl16
python -m aha encode --name dinov3_vitl16 --device cuda --ids 001 --limit 64 --review-dir review/gpu_smoke_encoder
bash scripts/run_server.sh features
```

`fetch_encoder.py` ghim commit SHA vào `assets/encoders/<name>/source.json`. Kiểm tra lại tên repo trên Hub trước khi tải; tôi chưa tải thử model này. Giai đoạn `features` chạy `verify`, test, audit, rồi sinh:

- `artifacts/features/dinov3_vitl16/`: 2048 chiều mỗi giây, đầu vào của model;
- `artifacts/features/surgformer/`: hidden + score SurgFormer, dùng cho `surgformer_index`, ablation `seen_backbone` và VLM comparator.

Extraction resume theo video. Nếu đổi encoder (ví dụ `facebook/dinov2-large`, không bị gate), dùng `--name` khác và tạo config mới với `features.name` và `model.visual_dim` tương ứng; ghi thành amendment.

## 5. Train cho abstract

Chạy một seed trước để đo VRAM và thời gian:

```bash
python -m aha train --config configs/full.json --out runs/full/seed_17 --seed 17 --device cuda
```

Forward–backward của CRF giữ khoảng `T × 12³` số thực cho mỗi bước scan; với ca dài nhất (khoảng 18.000 giây) ước tính vài GB. Nếu thiếu VRAM, giảm `train.max_questions` trước.

Sau đó:

```bash
bash scripts/run_server.sh abstract    # full, independent, structured_only, residual_only × 3 seed + baseline trên val
```

Script tự `--resume` run còn dở. Mỗi run ghi `config.json`, `run.json`, `history.json`, `last.pt`, `best.pt`, `best_val_predictions.json`. Checkpoint được chọn hoàn toàn bằng validation (trung bình supported_joint@5 của val gốc và val dẫn xuất).

## 6. Test hồi cứu

Ghi ma trận run đã chốt vào study log, rồi chạy **một lần**:

```bash
bash scripts/evaluate_test.sh
```

Script chấm mọi biến thể có run, trên cả QA gốc và QA dẫn xuất, paired với `surgformer_index`, thêm giải mã cascade và hai chẩn đoán cho `full`, so với PCJD lịch sử, rồi tổng hợp ba seed vào `runs/seed_summary.json`. Bảng cho abstract lấy từ file này: `supported_joint@5`, `joint@5`, `answer`, `grounding@5` của `full` so với `independent`, `structured_only`, `residual_only`, cascade, `surgformer_index`, `prior`.

## 7. Ablation (sau abstract)

```bash
bash scripts/run_server.sh ablation    # no_workflow, no_derived, no_hard_negative, seen_backbone
bash scripts/evaluate_test.sh          # chỉ sau khi đã ghi log; chạy lại sẽ chấm lại mọi biến thể
```

## 8. VLM comparator (tuỳ chọn)

Như v1: `requirements-vlm.txt`, tải Qwen3-VL-4B đúng revision trong `assets/models.lock.json`, rồi `aha vlm-predict` / `aha vlm-train`. Nhánh này cần `artifacts/features/surgformer` và chưa được chạy với weights thật.

## 9. Đổi protocol

`python -m aha derive` tái sinh QA dẫn xuất từ nhãn đã stage; `python -m aha text` tái sinh embedding. Mọi thay đổi về giả định, config hay code sau khi freeze phải thành version mới: sửa `PROTOCOL` trong `src/aha/common.py`, ghi amendment vào `proposal.md`, chạy test, rồi `python -m aha freeze`.
