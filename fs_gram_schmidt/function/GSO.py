import numpy as np
import pandas as pd
import scipy as sp
from .helper import *


def rank_features(feature_df, target, risk=1):
    X = np.array(feature_df)
    y = target.copy()

    features = np.array(feature_df.columns)
    real_feature_indices = np.arange(features.shape[0])

    # Initialise varibales
    ranked_features = [
    ]  # This will hold list of feature names sorted by their ranks

    selected_features = np.array(
        [], dtype=int
    )  # This will hold the list of indices or features as in real_feature_indices sorted by ranks

    # For the first feature selection

    feature_selection_risk = {
        # Mapping for probability relevancy for a feature when compared with previously selected features
    }

    X_tr = X.transpose()

    n_features, n_instances = X_tr.shape[0], X_tr.shape[1]

    cos_sqr_xy = cosine_sqr(X_tr, y)

    feature_with_least_angle = np.where(cos_sqr_xy == np.max(cos_sqr_xy))
    selected_features = np.append(selected_features, feature_with_least_angle)
    ranked_features.append(features[feature_with_least_angle])
    refined_features = np.delete(features, selected_features)
    real_feature_indices = real_feature_indices[
        ~np.isin(real_feature_indices, selected_features)]
    feature_selection_risk[0] = 0

    for i in range(n_features - 1):
        x_least = sp.matrix(X_tr[selected_features])

        rowmask = np.zeros(X_tr.shape[0]).astype(int)
        rowmask[selected_features] = 1
        xmask = np.repeat(rowmask, n_instances).reshape(-1, n_instances)

        X_tr_masked = np.ma.MaskedArray(X_tr, xmask)
        X_rem_features = X_tr_masked.compressed().reshape(
            -1, n_instances)  # Matrix of remaining feature vectors

        # Finding null space of selected features
        x_least_null_space = sp.linalg.null_space(x_least,
                                                  n_features - (i + 1))

        # Projecting remaining features onto nullspace of selected features
        X_rem_features_projected = project_on_nullspace(
            X_rem_features, x_least_null_space)

        # Projecting target onto nullspace of selected features
        y_proj = project_on_nullspace(y, x_least_null_space)

        # Calculation of cosine**2 for each feature with target vector
        cos_sqr_xy = cosine_sqr(X_rem_features_projected, y_proj)

        feature_with_least_angle = real_feature_indices[np.where(cos_sqr_xy == np.max(cos_sqr_xy))][0]

        selected_features = np.append(selected_features,
                                      feature_with_least_angle)

        real_feature_indices = real_feature_indices[
            ~np.isin(real_feature_indices, selected_features)]

        ranked_features.append(features[feature_with_least_angle])
        refined_features = np.delete(features, selected_features)

        space_dimension = x_least_null_space.shape[1] - (i + 1)
        cdf = round(CDF(np.max(cos_sqr_xy), space_dimension), 6)

        Pv = 1 - cdf
        feature_selection_risk[
            i + 1] = feature_selection_risk[i] + Pv * (
                1 - feature_selection_risk[i])

        if feature_selection_risk[i + 1] > risk:
            break

    return {'ranked_features': ranked_features, 'feature_selection_risk': feature_selection_risk}
