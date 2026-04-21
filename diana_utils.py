import torch
import torch.nn.functional as F
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
from collections import defaultdict

@torch.no_grad()
def calculate_source_centroids(model, source_loader, device, converter):
    """
    Calculate class-wise centroids for source domain.
    """
    model.eval()
    model = model.to(device)

    feature_sum = defaultdict(lambda: torch.zeros(model.SequenceModeling_output, device=device))
    count = defaultdict(int)

    for image_tensors, labels in source_loader:
        image_tensors = image_tensors.to(device)

        batch_size = image_tensors.size(0)
        dummy_text = torch.zeros(batch_size, dtype=torch.long, device=device)
        contextual_features, _ = model(image_tensors, text=(dummy_text,), is_train=False)

        for i, label_str in enumerate(labels):
            # Encode 1 string -> [1, max_len]
            encoded_labels, lengths = converter.encode([label_str])
            seq_len = lengths[0].item()

            # Lấy vector ký tự thực (bỏ padding phía sau)
            encoded_labels = encoded_labels[0][:seq_len]
            char_feats = contextual_features[i][:seq_len]

            for char_idx, char_feat in zip(encoded_labels, char_feats):
                idx_val = char_idx.item()
                if idx_val in [converter.dict["[PAD]"], converter.dict["[UNK]"]]:
                    continue
                char_token = converter.character[idx_val]
                feature_sum[char_token] += char_feat
                count[char_token] += 1

    centroids = {
        char: (feature_sum[char] / count[char]).detach().cpu()
        for char in feature_sum.keys() if count[char] > 0
    }

    return centroids


def compute_entropy(logits):
    """logits: [B, T, num_classes]"""
    probs = F.softmax(logits, dim=-1)                    # [B, T, C]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B, T]
    return entropy.mean(dim=1)  # [B] mean entropy per sequence


def compute_domainness(features, source_centroids):
    """features: [B, T, hidden_size], source_centroids: dict {char_id: centroid (d,)}"""
    centroids = torch.stack(list(source_centroids.values()))  # [K, d]
    # flatten features: [B*T, d]
    B, T, d = features.shape
    features_flat = features.view(B * T, d)

    dists = torch.cdist(features_flat, centroids)  # [B*T, K]
    min_dists, _ = dists.min(dim=1)                # [B*T]
    mean_dists = min_dists.view(B, T).mean(dim=1)  # [B]
    return mean_dists


def static_separate_subsets(model, target_loader, source_centroids, device, num_subsets=4):
    model.eval()
    all_uncertainties, all_domainnesses = [], []
    
    with torch.no_grad():
        for images in target_loader:  # unlabeled target data
            images = images.to(device)
            batch_size = images.size(0)
            dummy_text = torch.zeros(batch_size, dtype=torch.long, device=device)  # [SOS] token id giả

            contextual_feature, preds = model(images, text=(dummy_text,), is_train=False)  # [B, T, C]

            # --- batch compute ---
            batch_uncertainties = compute_entropy(preds)                  # [B]
            batch_domainnesses = compute_domainness(contextual_feature, source_centroids)  # [B]

            all_uncertainties.append(batch_uncertainties.cpu())
            all_domainnesses.append(batch_domainnesses.cpu())
    
    # concat toàn bộ
    U = torch.cat(all_uncertainties).numpy()
    D = torch.cat(all_domainnesses).numpy()

    # Normalize scores
    U = (U - U.min()) / (U.max() - U.min() + 1e-8)
    D = (D - D.min()) / (D.max() - D.min() + 1e-8)
    scores = np.stack([U, D], axis=1)  # [N, 2]
    
    # Cluster into subsets using GMM
    gmm = GaussianMixture(n_components=num_subsets, random_state=0).fit(scores)
    cluster_ids = gmm.predict(scores)  # [N]
    
    # Map clusters to cc/uc/ci/ui by mean values
    cluster_means = gmm.means_  # [4, 2] (uncertainty, domainness)
    labels_map = {}
    sorted_ids = np.argsort(cluster_means[:,0] + cluster_means[:,1])  # simple ordering
    for rank, cid in enumerate(sorted_ids):
        if rank == 0: labels_map[cid] = "cc"
        elif rank == 1: labels_map[cid] = "uc"
        elif rank == 2: labels_map[cid] = "ci"
        else: labels_map[cid] = "ui"
    
    subsets = [labels_map[cid] for cid in cluster_ids]
    return subsets, U, D

