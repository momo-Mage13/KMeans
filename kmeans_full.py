# -*- coding: utf-8 -*-
"""
基于 K-Means 的图像矢量量化
课程：Python 机器学习
实验内容：手写 K-Means + 肘击法选 K + 图像颜色压缩重建
"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第 1 部分：数据准备
# ============================================================
def load_image():
    im = Image.open('data/flower.jpg')
    image = np.array(im).astype(float) / 255     # 0~255 -> 0~1，颜色相对关系不变
    image = image[:, :, :3]                       # 只保留 R、G、B 三通道（去掉可能的 alpha）
    H, W, C = image.shape
    print("图片形状:", image.shape)                # (427, 640, 3)

    plt.figure(figsize=(8, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.title("原图（彩色）")
    plt.show()

    X = image.reshape(-1, 3)                      # (H*W, 3)：每个像素一条 [R,G,B] 样本
    print("样本矩阵:", X.shape)                   # (273280, 3)
    return image, X, H, W


# ============================================================
# 第 2 部分：手写 K-Means
# ============================================================
def initCenter(dataSet, k, random_state=42):
    """第一步：从数据集中随机选 k 个样本作为初始簇中心"""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(dataSet), size=k, replace=False)
    return dataSet[idx].copy()


def myKmeans(dataSet, k, random_state=42, max_iter=100):
    """
    K-Means 主循环：分配样本 -> 更新中心 -> 判断收敛

    返回:
        classCenter  : (k, 3)  最终的 k 个簇中心（即 k 个代表颜色）
        clusterPoints: (n, 2)  第 0 列：每个样本的簇编号；
                                第 1 列：样本到所属中心的距离平方；
                                第 1 列全部求和即为 SSE。
    """
    n = len(dataSet)
    centers = initCenter(dataSet, k, random_state)
    labels = np.full(n, -1, dtype=int)            # 上一轮的簇分配

    for _ in range(max_iter):
        # ---- 第二步：分配样本（向量化计算欧氏距离平方）----
        # dataSet: (n,1,3),  centers: (1,k,3)  ->  广播成 (n,k,3)
        dist = np.sum((dataSet[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1)

        # ---- 第四步：判断收敛（本轮簇分配与上一轮完全相同则停止）----
        if np.all(new_labels == labels):
            break
        labels = new_labels

        # ---- 第三步：更新中心 = 本簇所有样本的均值；空簇保留原中心 ----
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = dataSet[mask].mean(axis=0)

    # 收敛后再算一次最终距离，保证 clusterPoints 与最终中心严格对应
    dist = np.sum((dataSet[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist, axis=1)
    min_dist = dist[np.arange(n), labels]
    clusterPoints = np.column_stack([labels, min_dist])
    return centers, clusterPoints


# ============================================================
# 第 3 部分：肘击法确定最优 K
# ============================================================
def elbow(X, k_range=range(1, 17), seeds=(42, 7, 2024)):
    """
    K = 1..16，每个 K 换 2~3 个随机种子跑 myKmeans，
    取其中最小的 SSE 作为该 K 的代表值（反映这个 K 能达到的最好水平）。
    """
    sse_list = []
    for k in k_range:
        best_sse = np.inf
        for seed in seeds:
            _, cp = myKmeans(X, k, random_state=seed)
            sse = cp[:, 1].sum()
            if sse < best_sse:
                best_sse = sse
        sse_list.append(best_sse)
        print(f"K = {k:2d}   SSE = {best_sse:.4f}")

    sse_arr = np.array(sse_list)

    # 画 SSE-K 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), sse_arr, 'o-', color='#245bdb', linewidth=2, markersize=6)
    for k, s in zip(k_range, sse_arr):
        plt.annotate(f'{s:.1f}', (k, s), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)
    plt.xlabel('聚类数 K')
    plt.ylabel('SSE（簇内误差平方和）')
    plt.title('肘击法（Elbow Method）：SSE-K 曲线')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(list(k_range))
    plt.tight_layout()
    plt.savefig('elbow_curve.png', dpi=150)
    plt.show()
    return sse_arr


# ============================================================
# 第 4 部分：矢量量化重建
# ============================================================
def reconstruct(X, H, W, k, seed=42):
    """在全部像素上跑 K-Means，把每个像素换成所属簇的代表颜色，再 reshape 回图像"""
    centers, cp = myKmeans(X, k, random_state=seed)
    labels = cp[:, 0].astype(int)
    new_X = centers[labels]                     # 查表：每个像素 -> 代表颜色
    return new_X.reshape(H, W, 3), centers, cp[:, 1].sum()


def show_compare(image, H, W, X, best_k):
    img_k2,  _, sse2  = reconstruct(X, H, W, 2)
    img_best, c_best, sse_best = reconstruct(X, H, W, best_k)
    img_k30, _, sse30 = reconstruct(X, H, W, 30)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    titles = ['原图',
              'K = 2（2 种代表颜色）',
              f'K = {best_k}（最优 K）',
              'K = 30（30 种代表颜色）']
    imgs = [image, img_k2, img_best, img_k30]
    for ax, t, im_ in zip(axes, titles, imgs):
        ax.imshow(im_)
        ax.set_title(t, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('reconstruct_compare.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 压缩率：原图每像素 24 bit；重建图 = 每像素 log2(K) bit 索引 + K*24 bit 调色板
    orig_bits = H * W * 24
    print("\n=== 压缩率对比 ===")
    for k, sse in [(2, sse2), (best_k, sse_best), (30, sse30)]:
        total_bits = H * W * np.log2(k) + k * 24
        ratio = total_bits / orig_bits * 100
        print(f"K={k:2d}: SSE={sse:8.2f}  重建数据量={total_bits/8/1024:6.1f} KB  "
              f"占原图({orig_bits/8/1024:.1f} KB) 的 {ratio:.1f}%")


# ============================================================
# 主流程
# ============================================================
if __name__ == '__main__':
    image, X, H, W = load_image()

    print("\n=== 肘击法 ===")
    sse_arr = elbow(X)

    # 根据曲线观察，最优 K = 6（见思考题 1 的理由）
    BEST_K = 6
    print(f"\n>>> 选定最优 K = {BEST_K}")

    print("\n=== 矢量量化重建 ===")
    show_compare(image, H, W, X, BEST_K)
