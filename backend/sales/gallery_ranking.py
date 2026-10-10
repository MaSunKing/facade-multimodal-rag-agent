"""Rank eligible company assets by metadata scores, never inferred pixels."""

COMPANY_GALLERY_DISPLAY_LIMIT = 10


def rank_gallery_candidates(candidates, labels_for_asset):
    """Relevance first within a variant; distinct variants before repeats.

    Eligibility/ACL/exact identity belong to the caller. Scores reuse retrieval
    and label-text matching; this helper never runs an image or language model.
    """
    ranked = sorted(candidates, key=lambda item: (
        -float(item[0]), -int(item[1].get('display_priority') or 0),
        str(item[1].get('asset_id') or '')))
    unique, repeated, hashes, groups = [], [], set(), set()
    for item in ranked:
        asset = item[1]
        labels = labels_for_asset(asset.get('asset_id')) or {}
        digest = labels.get('sha256') or asset.get('asset_id')
        if digest in hashes:
            continue
        hashes.add(digest)
        projects=tuple(sorted(labels.get('project_names') or []))
        group = (labels.get('product_family_key'),
                 projects or labels.get('variant_or_code') or labels.get('product_name') or asset.get('asset_id'),
                 labels.get('visual_role') or asset.get('visual_role'))
        (repeated if group in groups else unique).append(item)
        groups.add(group)
    return unique + repeated
