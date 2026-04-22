# Tóm tắt paper: "Scalable Multi-Agent Reinforcement Learning for Warehouse Logistics with Robotic and Human Co-Workers"

**Tác giả:** Aleksandar Krnjaic, Raul D. Steleac, Jonathan D. Thomas, Georgios Papoudakis, Lukas Schäfer, Andrew Wing Keung To, Kuan-Ho Lao, Murat Cubuktepe, Matthew Haley, Peter Börsting, Stefano V. Albrecht
**Đơn vị:** Dematic (công ty logistics) + University of Edinburgh
**arXiv:** 2212.11498v3 (bản cập nhật 30/8/2024)

---

## 1. Bài toán

### 1.1 Order-picking

- Khách hàng gửi một "order" gồm nhiều **order-line** (mặt hàng + số lượng).
- **Pick rate** = số order-line giao được mỗi giờ — KPI chính của kho.
- Mục tiêu: tối đa pick rate = tối thiểu thời gian hoàn thành đơn.

### 1.2 Hai paradigm picking

| Paradigm | Cách hoạt động | Ví dụ thực tế |
|---|---|---|
| **PTG** (Person-to-Goods) | Picker (người) đi vòng quanh kho, AGV mang xe đẩy đi theo để chở hàng picker bốc. | Kho kiểu truyền thống có robot phụ trợ. |
| **GTP** (Goods-to-Person) | AGV chở kệ hàng đến **trạm picker cố định**; robot picker bốc hàng từ kệ, AGV mang kệ về. | Amazon Kiva, Quicktron QuickBin, Dematic Multishuttle. |

GTP đầu tư cao hơn nhưng throughput lớn hơn, khó mở rộng vì phụ thuộc layout cứng. Paper này giải quyết **cả 2 paradigm**.

### 1.3 Tại sao khó

- **Hành động heterogeneous:** AGV (chở kệ) và Picker (bốc hàng) có vai trò khác nhau, phải **phối hợp cùng lúc cùng chỗ** để thực hiện 1 pick.
- **Action space lớn:** mỗi agent có thể chọn nhiều vị trí kho khác nhau làm đích đến → không gian hành động tổ hợp bùng nổ.
- **Sparse reward:** giữa 2 lần giao thành công có hàng trăm step không reward.
- **Đa dạng cấu hình kho** (size, layout, số agent, tỉ lệ AGV:Picker) → heuristic tay phải tune lại mỗi lần.

---

## 2. Đóng góp chính của paper

1. **Hierarchical Multi-Agent RL (MARL)** cho order-picking:
   - Một **Manager agent** gán task (zone) cho từng worker.
   - Các **Worker agent** (AGV / Picker) chọn vị trí cụ thể trong zone được gán.
   - Manager + Worker được **train đồng thời** bằng MARL để max pick-rate.
2. Hierarchical MARL giảm mạnh action space hiệu dụng của mỗi agent → **sample efficiency cao hơn** MARL phẳng.
3. Vượt trội rõ so với heuristic công nghiệp và baseline MARL phẳng trên **nhiều cấu hình kho** và **cả 2 paradigm PTG + GTP**.
4. Mở source môi trường TA-RWARE làm benchmark cộng đồng (code bạn đang dùng).

---

## 3. Môi trường

### 3.1 Hai simulator

- **Dematic PTG simulator** (closed-source) — mô phỏng chính xác kho Dematic thật, dùng cho các thí nghiệm PTG.
- **TA-RWARE** (open-source) — mở rộng của RWARE, mô phỏng GTP kiểu Quicktron QuickBin. Đây là repo code bạn đang chạy.

### 3.2 Định nghĩa hình thức

Kho được biểu diễn bằng bộ 3 `W = {L, Z, W}`:
- `L` — tập vị trí trong kho: `L_item` (shelf), `L_delivery` (goal), `L_other` (nhàn rỗi).
- `Z` — phân phối order (do nhà cung cấp + hành vi khách hàng quyết định).
- `W = V ∪ P` — tập worker: `V` là AGV, `P` là Picker.

### 3.3 Action space

- Action của mỗi agent = **vị trí trong kho** để đi tới (không phải up/down/left/right).
- AGV: chọn bất kỳ `l ∈ L`.
- Picker: chọn bất kỳ `l ∈ L_item`.
- **Pathfinding tự động** bằng A\*, không phải học thấp cấp.
- **Invalid action masking**: ẩn các action không có ý nghĩa (ví dụ AGV đang cầm shelf không thể chọn shelf khác để pick).

### 3.4 Reward

- Pick hoặc deliver 1 item: **+1** (TA-RWARE) / **+0.1** (Dematic PTG).
- Penalty thời gian: **-0.001 mỗi step**.
- Reward **riêng cho từng agent** → POSG (Partially Observable Stochastic Game).

---

## 4. Phương pháp đề xuất — Hierarchical MARL

### 4.1 Kiến trúc 3 tầng

```
Manager   ── nhìn toàn cục ──→  gán zone y^i cho mỗi worker
    ↓
Worker i  ── nhìn local + zone được gán ──→  chọn vị trí cụ thể a^i trong zone
    ↓
Low-level controller ── A* ──→ sinh chuỗi primitive action (up/down/left/right)
```

### 4.2 Chia zone

Kho được chia thành **các zone không giao nhau** `Y`. Manager chọn 1 zone cho mỗi worker agent. Action space của manager = `|Y|^|workers|` — nhỏ hơn rất nhiều so với `|L|^|workers|` của phẳng.

### 4.3 Reward của manager

Manager nhận **sum reward từ các worker** mà nó đã gán goal. Một worker chỉ "đóng góp" reward cho manager trong khoảng thời gian từ khi nhận goal đến khi đạt goal.

### 4.4 Ba biến thể MARL áp dụng vào khung hierarchical

| Kí hiệu | Đầy đủ | Mô tả |
|---|---|---|
| **HIAC** | Hierarchical Independent Actor-Critic | Mỗi agent có actor + critic độc lập. |
| **HSNAC** | Hierarchical Shared-Network AC | Các picker chia sẻ 1 network, AGV chia sẻ 1 network khác. |
| **HSEAC** | Hierarchical Shared-Experience AC | Mỗi agent mạng riêng, nhưng chia sẻ gradient update cùng loại. |

Các phiên bản flat tương ứng (không hierarchical) là: **IAC**, **SNAC**, **SEAC**.

---

## 5. Heuristic baselines

### 5.1 Follow Me (FM) — cho PTG

Nhiều AGV đi cùng 1 Picker (thành một nhóm). Picker đi theo thứ tự do TSP tính sẵn. AGV không rời picker cho tới khi nhóm làm xong toàn bộ order. Ưu: picker không idle. Nhược: AGV có thể đi nhiều hơn cần thiết.

### 5.2 Pick, Don't Move (PDM) — cho PTG

Picker được chia zone cố định, không rời. AGV đi đến vị trí item theo TSP. Khi AGV đến zone, picker phục vụ AGV nào gần nhất. Ngược với FM: picker có thể idle khi đơn trong zone ít.

### 5.3 Closest Task Assignment (CTA) — cho GTP (= heuristic trong `tarware/heuristic.py`)

- AGV gần nhất nhận order đầu trong queue (dùng A\*-distance).
- Picker chia zone tĩnh, phục vụ AGV theo FIFO order.
- AGV đem kệ về goal gần nhất, rồi trả kệ về ô trống gần nhất.

---

## 6. Kết quả thực nghiệm

### 6.1 Setup

- Train **10,000 episode** mỗi run, 5 seed, báo cáo **trung bình 50 episode cuối** ± 95% CI.
- So cả **hierarchical (HIAC/HSNAC/HSEAC)** vs **flat (IAC/SNAC/SEAC)** vs **heuristic (FM/PDM/CTA)**.

### 6.2 Bảng kết quả chính (pick rate = order-lines/giờ)

#### Dematic PTG simulator

| Method | Small | Medium | Large | Disjoint |
|---|---|---|---|---|
| FM | 901.3 | 1098.1 | 1230.2 | 568.4 |
| PDM | 783.6 | 982.2 | 1123.9 | 677.4 |
| IAC (flat) | 1053.0 | 1206.4 | 1263.9 | 733.2 |
| SEAC (flat) | 1019.7 | 1185.1 | 1262.9 | 739.8 |
| **HSEAC (ours)** | **1028.2** | **1242.1** | **1370.9** | **803.5** |

→ HSEAC vượt FM: +13.1% (Medium), +11.4% (Large), **+41.3% (Disjoint)**.
→ HSEAC vượt PDM: +26.5%, +22.0%, +18.6%.

#### TA-RWARE (GTP) — môi trường open-source

| Method | Small | Large |
|---|---|---|
| CTA (heuristic) | 52.7 | 67.1 |
| IAC (flat) | 65.2 | 80.4 |
| SEAC (flat) | 64.8 | 82.2 |
| **HIAC (ours)** | **66.7** | **86.0** |
| HSNAC (ours) | 66.0 | 85.0 |
| HSEAC (ours) | 64.6 | 84.8 |

→ HIAC vượt CTA: **+26.6% (Small)**, **+28.2% (Large)**.

### 6.3 Nhận xét

1. **MARL luôn thắng heuristic** ở mọi cấu hình. Khoảng cách lớn nhất ở kho phức tạp (Disjoint: +41%).
2. **Hierarchical > Flat**, đặc biệt khi kho scale lớn — do giảm action space.
3. **SNAC phẳng (shared network)** tệ nhất — mọi agent dùng chung policy → dễ deadlock. HSNAC thêm manager phân vùng → fix được.
4. **Hiệu quả sample efficiency**: Hierarchical đạt performance cao chỉ sau ~2000 episode, flat cần >5000.

---

## 7. Kết luận & hạn chế

### 7.1 Điểm mạnh

- Phương pháp tổng quát, scale được theo kích thước kho và số worker.
- Mở rộng sang cả PTG và GTP không cần sửa đổi cấu trúc.
- Thắng heuristic công nghiệp **nhiều phần trăm** trên KPI thật sự quan trọng (pick rate).

### 7.2 Hạn chế / future work

- Chỉ tối ưu pick rate — chưa xét travel distance, energy, welfare của nhân viên.
- Chưa tự động chia zone — `Y` vẫn được định nghĩa thủ công (chia đều). Tương lai: dùng **unsupervised environment design** để học phân vùng tối ưu.
- Chưa xét **sub-task decomposition** tự động.

### 7.3 Liên hệ thực tiễn

Dematic là công ty thật có kho thật. Paper là bước đầu cho việc deploy MARL cho order-picking **ngoài production**. Với kho 24/7, cải thiện +28% pick-rate là giá trị thương mại khổng lồ, không cần đầu tư phần cứng thêm.

---

## 8. Liên hệ với code seminar

### 8.1 Code repo `task-assignment-robotic-warehouse`

Chỉ chứa:
- **Môi trường TA-RWARE** (GTP).
- **Heuristic CTA** (baseline so sánh).

Không chứa:
- Phần **hierarchical MARL** (HIAC/HSNAC/HSEAC) — đóng góp chính của paper, không public.
- Dematic PTG simulator (IP công ty).

### 8.2 Tái lập phần MARL

Có thể dùng **EPyMARL** (cùng nhóm tác giả viết) để tái lập các đường **flat** trong Hình 4:
- IAC → `ia2c` trong EPyMARL.
- MAPPO (tương đương tinh thần CTDE, không có trong paper gốc nhưng là baseline phổ biến).

Phần hierarchical không có implementation public, cần tự viết từ zero theo mô tả ở Section IV-B của paper nếu muốn reproduce.

---

## 9. Tham khảo nhanh

- **Paper PDF:** `Paper/2212.11498v3.pdf`
- **Code môi trường:** https://github.com/uoe-agents/task-assignment-robotic-warehouse
- **EPyMARL (MARL baselines):** https://github.com/uoe-agents/epymarl
- **RWARE gốc:** https://github.com/uoe-agents/robotic-warehouse
