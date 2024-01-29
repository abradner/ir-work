import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from numpy import percentile

from signal_processing import extract_pulses_and_spaces

from gather_raw_codes import RawCodeGatherer


# def cluster_samples(samples):
#     """Cluster samples into two clusters."""
#     samples = np.array(samples).reshape(-1, 1)
#     kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(samples)
#     averages = [np.mean(samples[kmeans.labels_ == i]) for i in range(2)]
#     averages.sort()
#     return averages[0], averages[1]

# def sample_confidence(samples):
#     """Calculate the confidence of the samples."""
#     samples = np.array(samples).reshape(-1, 1)
#     kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(samples)
#     score = silhouette_score(samples, kmeans.labels_)
#     return score

def cluster_samples(samples, method='kmeans'):
    """Cluster samples into two clusters."""
    samples = np.array(samples).reshape(-1, 1)
    if method == 'kmeans':
        model = KMeans(n_clusters=2, random_state=0, n_init=10)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=2)
    elif method == 'meanshift':
        model = MeanShift(bandwidth=2)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit(samples)
    averages = [np.mean(samples[model.labels_ == i]) for i in range(2)]
    averages.sort()
    return averages[0], averages[1]

def sample_confidence(samples, method='kmeans'):
    """Calculate the confidence of the samples."""
    samples = np.array(samples).reshape(-1, 1)
    if method == 'kmeans':
        model = KMeans(n_clusters=2, random_state=0, n_init=10)
    elif method == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=2)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=2)
    elif method == 'meanshift':
        model = MeanShift(bandwidth=2)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    model.fit(samples)
    score = silhouette_score(samples, model.labels_)
    return score

def get_cluster_arrays(samples):
    """Get the cluster arrays."""
    samples = np.array(samples).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(samples)
    distances = kmeans.transform(samples)
    return distances

def filter_outliers(samples, method='zscore'):
    """Filter outliers from the samples."""
    if method == 'zscore':
        z_scores = zscore(samples)
        filtered_samples = [s for s, z in zip(samples, z_scores) if abs(z) < 3]
    elif method == 'iqr':
        Q1 = percentile(samples, 25)
        Q3 = percentile(samples, 75)
        IQR = Q3 - Q1
        filtered_samples = [s for s in samples if Q1 - 1.5 * IQR <= s <= Q3 + 1.5 * IQR]
    elif method == 'isolation_forest':
        model = IsolationForest(contamination=0.1)
        preds = model.fit_predict(np.array(samples).reshape(-1, 1))
        filtered_samples = [s for s, p in zip(samples, preds) if p == 1]
    else:
        raise ValueError(f"Unknown method: {method}")
    return filtered_samples

def tune_ir_structure(gatherer, clustering_method='kmeans', outlier_method='zscore'):
    """Tune the IR structure using the pulses and spaces."""
    
    pulse_confidence = 0
    space_confidence = 0
    raw_signal = []
    min_samples = 32
    
    while pulse_confidence < 0.95 or space_confidence < 0.95 or len(raw_signal) < min_samples:
        line = gatherer.gather_signal(1)
        raw_signal += line

        if len(raw_signal) < min_samples:
            print(f"Warning: Not enough samples. Sample count: {len(raw_signal)}")
            continue

        pulses, spaces = extract_pulses_and_spaces(raw_signal)

        # exclude header pulse and space
        pulses = filter_outliers(pulses[1:], method=outlier_method)
        spaces = filter_outliers(spaces[1:], method=outlier_method)

        pulse_confidence = sample_confidence(pulses, method=clustering_method) 
        space_confidence = sample_confidence(spaces, method=clustering_method)
        print(f"Pulse confidence: {pulse_confidence}, space confidence: {space_confidence}")

    header_pulse, _ = max(pulses), min(pulses)
    header_space, _ = max(spaces), min(spaces)

    # Exclude header pulse and space
    pulses = pulses[1:]
    spaces = spaces[1:]

    zero_pulse, one_pulse = cluster_samples(pulses, method=clustering_method)
    zero_space, one_space = cluster_samples(spaces, method=clustering_method)
    
    tolerance = 100

    tuning_data = {
            "header_pulse": header_pulse,
            "header_space": header_space,
            "zero_pulse": zero_pulse,
            "zero_space": zero_space,
            "one_pulse": one_pulse,
            "one_space": one_space,
            "tolerance": tolerance,
        }
    

    return tuning_data
