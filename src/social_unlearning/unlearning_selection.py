"""Validation gates shared by backbone adapters; no test metrics are consumed.

Candidate mass is a mechanistic diagnostic, not a privacy/erasure guarantee.
A passing checkpoint still needs independent membership/forgetting evaluation.
"""
import math


def assess_candidate(metrics, initial, reference, masses, initial_masses, *,
                     utility_floor=.98, mass_reduction=.01):
    if not 0 < utility_floor <= 1 or not 0 <= mass_reduction < 1:
        raise ValueError('Invalid validation thresholds')
    keys = ('interaction_deleted_candidate_mass', 'social_deleted_candidate_mass')
    values = [metrics[k] for k in ('recall@20', 'ndcg@20')]
    values += [metrics['fidelity'][k] for k in ('score_rmse', 'top20_overlap')]
    values += [reference[k] for k in ('recall@20', 'ndcg@20')]
    values += [initial['fidelity'][k] for k in ('score_rmse', 'top20_overlap')]
    values += [m[k] for m in (masses, initial_masses) for k in keys]
    finite = all(math.isfinite(float(v)) for v in values)
    utility = finite and all(metrics[k] >= utility_floor * max(reference[k], initial[k])
                             for k in ('recall@20', 'ndcg@20'))
    fidelity = finite and (metrics['fidelity']['score_rmse'] <= initial['fidelity']['score_rmse']
                           and metrics['fidelity']['top20_overlap'] >= initial['fidelity']['top20_overlap'])
    # Strict improvement excludes unchanged initialization and zero-mass cases.
    mass = finite and all(initial_masses[k] > 0 and masses[k] < initial_masses[k]
                          and masses[k] <= (1 - mass_reduction) * initial_masses[k] for k in keys)
    return dict(finite=finite, utility_passed=utility, fidelity_passed=fidelity,
                mass_proxy_passed=mass, eligible=utility and fidelity and mass,
                limitation='Mass reduction is not an erasure or membership guarantee')


def candidate_rank(metrics, step):
    """Lowest validation RMSE, then highest overlap, then fewer updates."""
    f = metrics['fidelity']
    return (f['score_rmse'], -f['top20_overlap'], step)
