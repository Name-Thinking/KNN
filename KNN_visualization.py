import os
import urllib.request
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# =========================================================
# 0. จัดการฟอนต์ภาษาไทย (เปลี่ยนวิธีดาวน์โหลดและเช็คไฟล์)
# =========================================================
FONT_PATH = "thsarabunnew.ttf"
if not os.path.exists(FONT_PATH):
    FONT_URL = "https://github.com/Phonbopit/sarabun-webfont/raw/master/fonts/thsarabunnew-webfont.ttf"
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)

fm.fontManager.addfont(FONT_PATH)
plt.rcParams.update(
    {
        "font.family": "TH Sarabun New",
        "axes.unicode_minus": False,
        "font.size": 14,
    }
)


# =========================================================
# 1. Custom KNN Classifier (เขียนแบบ Vectorized operations)
# =========================================================
class KNearestNeighbors:

    def __init__(self, metric="euclidean"):
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.int64)
        return self

    def _calc_distances(self, target_point):
        """คำนวณระยะทางจาก target ไปยังทุกจุดใน X_train พร้อมกัน"""
        if self.metric == "manhattan":
            return np.sum(np.abs(self.X_train - target_point), axis=1)
        return np.linalg.norm(self.X_train - target_point, axis=1)

    def predict(self, X_test, k):
        X_test = np.atleast_2d(X_test)
        preds = []
        for test_pt in X_test:
            dists = self._calc_distances(test_pt)
            top_k_indices = np.argsort(dists)[:k]
            top_k_labels = self.y_train[top_k_indices]

            # ใช้ np.bincount หาค่าที่เกิดบ่อยสุดแทน Counter
            most_frequent = np.bincount(top_k_labels).argmax()
            preds.append(most_frequent)
        return np.array(preds)

    def get_k_neighbors_info(self, target_point, k):
        dists = self._calc_distances(target_point)
        sorted_indices = np.argsort(dists)[:k]
        return sorted_indices, dists[sorted_indices]


# =========================================================
# 2. ข้อมูลระบบและค่าเริ่มต้น
# =========================================================
X_dataset = np.array(
    [
        [1, 4],
        [2, 5],
        [3, 6],  # Class 0
        [5, 5],
        [6, 4],
        [7, 3],  # Class 1
        [2, 2],
        [1, 1],  # Class 2
        [4, 0],
    ]
)

y_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
query_pt = np.array([3.5, 3.5])

# กำหนดสเปกของแต่ละ Class
LABELS_CONFIG = {
    0: {"name": "สีน้ำเงิน", "color": "royalblue"},
    1: {"name": "สีเขียว", "color": "mediumseagreen"},
    2: {"name": "สีแดง", "color": "tomato"},
}

# =========================================================
# 3. รับค่าและประมวลผล
# =========================================================
print("กำหนดค่า k (เลือกค่าจำนวนเต็มระหว่าง 1 ถึง 9):")
user_k_list = [
    int(input(f"กรอกค่า k ตัวที่ {i+1}: ")) for i in range(3)
]

clf = KNearestNeighbors(metric="euclidean").fit(X_dataset, y_labels)

print("\n=== ผลการทำนายด้วย KNN ===")
for k_val in user_k_list:
    res = clf.predict(query_pt, k=k_val)[0]
    print(
        f"เมื่อ k={k_val} → จุด X จัดอยู่ในประเภท: {LABELS_CONFIG[res]['name']}"
    )


# =========================================================
# 4. วาดภาพและกราฟแสดงผล
# =========================================================
fig, ax = plt.subplots(figsize=(6.5, 6.5))

# พล็อตจุดฝึกสอนแยกตามคลาส
for c_id, config in LABELS_CONFIG.items():
    pts = X_dataset[y_labels == c_id]
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=config["color"],
        s=90,
        label=f"Class {c_id}",
        zorder=3,
    )

# พล็อตจุด Query
ax.scatter(
    query_pt[0],
    query_pt[1],
    c="black",
    marker="x",
    s=110,
    linewidths=1.8,
    label="จุด X",
    zorder=5,
)

# จัดกลุ่มวงกลมที่มีรัศมีเท่ากัน
radius_map = {}
for k in user_k_list:
    _, neighbor_dists = clf.get_k_neighbors_info(query_pt, k)
    r = round(neighbor_dists[-1], 4)
    radius_map.setdefault(r, []).append(str(k))

y_shifts = [0.35, -0.35, 0.85]

for idx, (r_val, k_group) in enumerate(sorted(radius_map.items())):
    label_text = f"k={', '.join(sorted(k_group, key=int))}"

    # วาดวงกลมขอบเขต
    circ = plt.Circle(
        query_pt,
        r_val,
        fill=False,
        ls="--",
        color="black",
        lw=1,
        alpha=0.85,
        zorder=2,
    )
    ax.add_patch(circ)

    # คำนวณจุดวางป้ายข้อความ
    dy = y_shifts[idx % len(y_shifts)]
    dx = np.sqrt(max(0, r_val**2 - dy**2))

    ax.text(
        query_pt[0] + dx + 0.08,
        query_pt[1] + dy,
        label_text,
        color="black",
        fontsize=12,
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="#cccccc",
            lw=0.8,
            alpha=0.85,
        ),
        zorder=6,
    )

# ตั้งค่าดิสเพลย์ของกราฟ
ax.set_title("ตัวอย่างการทำงานของ K-Nearest Neighbor", fontsize=16)
ax.set_xlabel("คุณสมบัติ 1", fontsize=14)
ax.set_ylabel("คุณสมบัติ 2", fontsize=14)
ax.set_xlim(-0.5, 8.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect("equal")
ax.grid(True, ls=":", alpha=0.6)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)

plt.tight_layout()
plt.show()