def calculate_suitability(user_rank, cutoff_rank, max_gap=60000):
    rank_gap = cutoff_rank - user_rank
    if rank_gap < 0:
        return 0.0
    gap_ratio = min(rank_gap / max_gap, 1)
    suitability = 85 * (1 - gap_ratio) + 15
    return round(min(max(suitability, 5), 95), 2)


def classify_risk(rank_gap):
    if rank_gap <= 2000:
        return "🎯 Ambitious"
    elif rank_gap <= 10000:
        return "⚖️ Moderate"
    else:
        return "✅ Safe"
