"""Estimate one global metric scale without letting isolated jumps dominate it."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .colmap_io import make_image_pose


@dataclass(frozen=True)
class ScaleFitOptions:
    filter_position_outliers: bool = True
    min_position_error_m: float = 0.15
    mad_factor: float = 6.0
    max_interval_stretch: float = 3.0
    max_iterations: int = 5
    max_excluded_fraction: float = 0.2
    max_relative_rmse: float = 0.05


def _match_distances(images, distances):
    if isinstance(images, dict):
        images = [make_image_pose(pose["image_id"], pose["camera_id"], name,
                                  pose["qvec"], pose["tvec"])
                  for name, pose in images.items()]
    by_basename = {}
    for name, distance in distances.items():
        by_basename.setdefault(name.rsplit("/", 1)[-1], []).append((name, distance))
    matched, used = [], set()
    for image in images:
        normalized = image.name.replace("\\", "/")
        if normalized in distances:
            distance_name, distance = normalized, distances[normalized]
        else:
            candidates = by_basename.get(normalized.rsplit("/", 1)[-1], [])
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous distance for image: {image.name}")
            distance_name, distance = candidates[0]
        if distance_name in used:
            raise ValueError(f"Distance record matches multiple cameras: {distance_name}")
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid distance for image: {image.name}")
        if not np.isfinite(image.center).all():
            raise ValueError(f"Nonfinite camera center for image: {image.name}")
        used.add(distance_name)
        matched.append((distance, image))
    if len(matched) < 2:
        raise ValueError("At least two registered front cameras with distances are required")
    matched.sort(key=lambda item: (item[0], item[1].name))
    return matched, len(distances) - len(used)


def _path_fit(records):
    centers = np.array([image.center for _, image in records])
    readings = np.array([distance for distance, _ in records])
    model_path = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(centers, axis=0), axis=1))]
    meter_path = readings - readings[0]
    if model_path[-1] <= 1e-12 or meter_path[-1] <= 0:
        raise ValueError("Camera trajectory and distance range must both be nonzero")
    scale = float(np.dot(model_path, meter_path) / np.dot(model_path, model_path))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated scale must be finite and > 0")
    residuals = scale * model_path - meter_path
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return scale, {
        "scale_meters_per_model_unit": scale,
        "first_image": records[0][1].name, "last_image": records[-1][1].name,
        "first_distance_m": float(readings[0]), "last_distance_m": float(readings[-1]),
        "model_path_length": float(model_path[-1]),
        "scaled_path_length_m": float(scale * model_path[-1]),
        "distance_span_m": float(meter_path[-1]),
        "fit_rmse_m": rmse,
        "fit_relative_rmse": rmse / float(meter_path[-1]),
        "fit_max_abs_error_m": float(np.max(np.abs(residuals))),
    }


def _filter_position_spikes(records, options):
    """Use interval medians before cumulative sums; never edit input poses.

    A candidate must have excessive incident chord lengths AND a local peak in
    its distance-interpolated position error. The chord condition protects real
    turns from being classified as spikes solely because they are not straight.
    Endpoint residuals use extrapolation from the next/previous two anchors.
    """
    retained, removed = list(records), []
    for iteration in range(options.max_iterations):
        if len(retained) < 5:
            break  # Too few samples for a reliable majority-based decision.
        readings = np.array([distance for distance, _ in retained])
        centers = np.array([image.center for _, image in retained])
        meter_steps = np.diff(readings)
        model_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        valid = (meter_steps > 1e-9) & (model_steps > 1e-12)
        if np.count_nonzero(valid) < 3:
            break
        preliminary_scale = float(np.median(meter_steps[valid] / model_steps[valid]))
        if not math.isfinite(preliminary_scale) or preliminary_scale <= 0:
            raise ValueError("Cannot obtain a finite positive scale from camera intervals")
        stretches = np.zeros(len(model_steps))
        stretches[valid] = preliminary_scale * model_steps[valid] / meter_steps[valid]
        errors = np.full(len(retained), np.nan)
        for index in range(len(retained)):
            first, second = ((1, 2) if index == 0 else
                             (index - 2, index - 1) if index == len(retained) - 1 else
                             (index - 1, index + 1))
            span = readings[second] - readings[first]
            if span <= 1e-9:
                continue
            alpha = (readings[index] - readings[first]) / span
            predicted = (1 - alpha) * centers[first] + alpha * centers[second]
            errors[index] = preliminary_scale * np.linalg.norm(centers[index] - predicted)
        finite_errors = errors[np.isfinite(errors)]
        if not len(finite_errors):
            break
        median = float(np.median(finite_errors))
        mad = float(np.median(np.abs(finite_errors - median)))
        limit = max(options.min_position_error_m, median + options.mad_factor * 1.4826 * mad)
        scores = np.where(np.isfinite(errors), errors, -np.inf)
        candidates = []
        for index, error in enumerate(scores):
            incident = stretches[max(0, index - 1):min(len(stretches), index + 1)]
            if (error > limit and np.all(incident > options.max_interval_stretch)
                    and (index == 0 or error >= scores[index - 1])
                    and (index == len(scores) - 1 or error >= scores[index + 1])):
                candidates.append(index)
        if not candidates:
            break
        if (len(removed) + len(candidates) > len(records) * options.max_excluded_fraction
                or len(retained) - len(candidates) < 3):
            raise ValueError("Too many inconsistent camera positions to estimate a reliable metric scale")
        for index in candidates:
            distance, image = retained[index]
            removed.append({
                "image_id": image.image_id, "image": image.name, "distance_m": distance,
                "camera_center_input": image.center.tolist(), "iteration": iteration + 1,
                "position_error_m": float(errors[index]), "position_threshold_m": limit,
                "preliminary_meters_per_input_unit": preliminary_scale,
            })
        rejected = set(candidates)
        retained = [record for index, record in enumerate(retained) if index not in rejected]
    return retained, removed


def estimate_metric_scale(images, distances: dict[str, float], *,
                          options: ScaleFitOptions | None = None) -> tuple[float, dict]:
    """Fit cumulative chords after removing isolated, gross position spikes.

    A 5% distance-span-normalized RMSE limit rejects unusable fits before COLMAP
    is called. This is a fit-quality check, not proof of ground-truth accuracy.
    All original poses remain available for the common Sim3 transformation.
    """
    options = ScaleFitOptions() if options is None else options
    positive = (options.min_position_error_m, options.mad_factor,
                options.max_interval_stretch, options.max_relative_rmse)
    if (any(not math.isfinite(value) or value <= 0 for value in positive)
            or not isinstance(options.max_iterations, int) or options.max_iterations < 1
            or not 0 < options.max_excluded_fraction < 0.5):
        raise ValueError("Invalid scale-fit thresholds or iteration count")
    matched, unused_count = _match_distances(images, distances)
    _, unfiltered = _path_fit(matched)
    retained, removed = (_filter_position_spikes(matched, options)
                         if options.filter_position_outliers else (matched, []))
    scale, report = _path_fit(retained)
    if report["fit_relative_rmse"] > options.max_relative_rmse:
        raise ValueError(
            "Metric scale fit is unreliable: "
            f"RMSE={report['fit_rmse_m']:.6g} m over {report['distance_span_m']:.6g} m "
            f"({report['fit_relative_rmse']:.2%}; limit {options.max_relative_rmse:.2%}). "
            "Check camera positions and cumulative distance records before stitching.")
    report.update({
        "matched_images": len(matched), "fit_images": len(retained),
        "unused_distance_records": unused_count,
        "fit_image_names": [image.name for _, image in retained],
        "excluded_pose_outliers": removed,
        "unfiltered_fit": unfiltered, "fit_options": asdict(options),
        "fit_quality_passed": True,
        "method": ("least_squares_cumulative_camera_path_chords_with_position_spike_filter"
                   if options.filter_position_outliers else "least_squares_cumulative_camera_path_chords"),
    })
    return scale, report
