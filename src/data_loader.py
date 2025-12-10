import logging
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from threading import Thread, Lock
from typing import Dict, Any, Optional, Tuple, Union
import time
import traceback

# Set up logging - will be configured by calling script
logger = logging.getLogger(__name__)

class UltraLowMemoryLoader(Dataset):
    # Cache class-balanced selections so repeated epochs don't recompute heavy logic
    _BALANCED_SELECTION_CACHE: Dict[str, Dict[str, Any]] = {}

    def __init__(self, data_root: str, split: str = "train", 
                 max_events_per_sample: int = 10000, annotation_dir: str = None,
                 max_samples_per_file: int = None, targeted_training: bool = True,
                 force_cpu: bool = False,
                 num_classes: int = 3, use_class_balanced_sampling: bool = False,
                 min_samples_per_class: int = 1, max_annotations_per_class: int = None,
                 cache_samples: bool = False, preload_all_samples: bool = False, debug_sample_loading: bool = False,
                 time_steps: int = 8, image_height: int = 720, image_width: int = 1280, time_window_us: int = 100000,):
        '''
        Initialize the UltraLowMemoryLoader.

        Args:
            data_root: The root directory of the data.
            split: The split of the data.
            max_events_per_sample: The maximum number of events per sample.
            annotation_dir: The directory of the annotations.
            max_samples_per_file: The maximum number of samples per file.
            targeted_training: Whether to use targeted training.
            force_cpu: Whether to force the use of CPU.
            use_3_class_annotations: Whether to use 3-class annotations.
            use_class_balanced_sampling: Whether to use class-balanced sampling.
            min_samples_per_class: The minimum number of samples per class.
            max_annotations_per_class: The maximum number of annotations per class.
            cache_samples: Whether to cache the samples.
            preload_all_samples: Whether to preload all samples into RAM.
            debug_sample_loading: Whether to debug the sample loading.
            time_steps: The number of time steps.
            image_height: The height of the image.
            image_width: The width of the image.
            time_window_us: The time window in microseconds.
        '''

        logger.info("=== INITIALIZING ULTRA LOW MEMORY LOADER ===")
        logger.info(f"Data root: {data_root}")
        logger.info(f"Split: {split}")
        logger.info(f"Max events per sample: {max_events_per_sample}")
        logger.info(f"Max samples per file: {max_samples_per_file}")
        logger.info(f"Max annotations per class: {max_annotations_per_class}")
        logger.info(f"Targeted training: {targeted_training}")
        logger.info(f"Annotation dir: {annotation_dir}")
        logger.info(f"Number of classes: {num_classes}")

        self.data_root = Path(data_root)
        self.split = split
        self.max_events_per_sample = max_events_per_sample
        self.max_samples_per_file = max_samples_per_file
        self.max_annotations_per_class = max_annotations_per_class  # Direct class balance limit
        self.targeted_training = targeted_training
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self.force_cpu = force_cpu  # Only force CPU tensor creation if explicitly requested
        # Unified device selection: CUDA if available and not forced to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu')
        self.num_classes = int(num_classes)  # Number of classes (dynamically determined from config)
        self.use_class_balanced_sampling = use_class_balanced_sampling  # Enable class-balanced sampling
        self.min_samples_per_class = min_samples_per_class  # Minimum samples per class
        self.cache_samples = cache_samples
        self.preload_all_samples = preload_all_samples
        self.debug_sample_loading = debug_sample_loading
        self._sample_cache: Optional[Dict[Tuple[int, int], Dict[str, torch.Tensor]]] = {} if cache_samples else None
        self.time_steps = int(time_steps)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.time_window_us = time_window_us

        # Add annotation caching to speed up repeated accesses
        self._annotation_cache = {}

        # Add precomputed event indices cache for fast targeted training
        self._event_indices_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}  # (file_idx, sample_idx) -> (start_idx, end_idx)

        # Find all HDF5 files
        self.h5_files = self._find_h5_files()
        logger.info(f"Found {len(self.h5_files)} HDF5 files")

        # Validate that files were found
        if len(self.h5_files) == 0:
            error_msg = (
                f"No HDF5 files found for split '{self.split}' in data root: {self.data_root}\n"
                f"Expected directory structure:\n"
                f"  - For 'train' split: subdirectories named 'train_*' (e.g., train_h5_1/, train_h5_2/)\n"
                f"  - For 'val' split: subdirectories named 'val_*' (e.g., val_h5_1/)\n"
                f"  - For 'test' split: subdirectories named 'test_*' (e.g., test_h5_1/)\n"
                f"Please check:\n"
                f"  1. Data root path is correct: {self.data_root}\n"
                f"  2. Subdirectories exist and match the split name\n"
                f"  3. HDF5 files (*.h5) exist in those subdirectories"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"[UltraLowMemoryLoader] Using device: {self.device} (force_cpu={self.force_cpu})")

        # Calculate dynamic samples per file based on actual event count
        if self.targeted_training:
            logger.info("Using targeted training: Only sampling time windows with annotations")
            logger.info("About to find annotated windows...")
            self.annotated_windows = self._find_annotated_windows()
            logger.info(f"Found annotated windows in {len(self.annotated_windows)} files")
            logger.info("Finished finding annotated windows, now filtering samples...")
            self.file_samples = self._filter_samples_by_annotations()
            logger.info("Finished filtering samples")
        else:
            logger.info("Using standard training: Sampling all time windows")
            self.file_samples = self._calculate_samples_per_file()

        # Create mapping from sample index to file and sample within file
        self.sample_mapping = self._create_sample_mapping()
        self.total_samples = len(self.sample_mapping)

        logger.info(f"Created {self.total_samples} samples with dynamic allocation:")
        for file_path, samples in self.file_samples.items():
            logger.info(f"  {file_path.name}: {samples} samples")

        # Validate that samples were created
        if self.total_samples == 0:
            error_msg = (
                f"No samples created for split '{self.split}'!\n"
                f"Found {len(self.h5_files)} HDF5 files but created 0 samples.\n"
            )
            if self.targeted_training:
                error_msg += (
                    f"Targeted training is enabled, but no annotated windows were found.\n"
                    f"Please check:\n"
                    f"  1. Annotation files (*_bbox.npy) exist for the HDF5 files\n"
                    f"  2. Annotation directory is correct: {self.annotation_dir}\n"
                    f"  3. Annotations are in the correct format\n"
                    f"  4. Files with annotations: {len(self.annotated_windows)}/{len(self.h5_files)}"
                )
            else:
                error_msg += (
                    f"Standard training is enabled, but no samples could be calculated.\n"
                    f"Please check:\n"
                    f"  1. HDF5 files contain event data\n"
                    f"  2. max_events_per_sample setting: {self.max_events_per_sample}\n"
                    f"  3. max_samples_per_file setting: {self.max_samples_per_file}"
                )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # PRE-LOAD ALL SAMPLES INTO RAM if requested (massive speedup!)
        if self.preload_all_samples and self.cache_samples:
            logger.info(f"\n{'='*60}")
            logger.info(f"PRE-LOADING ALL {self.total_samples} SAMPLES INTO RAM...")
            logger.info(f"This will take a few minutes but make training MUCH faster!")
            logger.info(f"{'='*60}\n")
            self._preload_all_windows()
            logger.info(f"\n{'='*60}")
            logger.info(f"PRE-LOADING COMPLETE! All samples cached in RAM.")
            logger.info(f"{'='*60}\n")
        elif self.preload_all_samples and not self.cache_samples:
            logger.warning("preload_all_samples=True but cache_samples=False. Preloading disabled.")

        logger.info("=== ULTRA LOW MEMORY LOADER INITIALIZATION COMPLETE ===")

    def _find_h5_files(self):
        """Find all HDF5 files, filtering by split (train/val/test) based on PARENT DIRECTORY NAME, not filename.

        Files are organized in subdirectories like:
        - train_h5_1/, train_h5_2/, ... (for training)
        - val_h5_1/, val_h5_2/, ... (for validation)
        - test_h5_1/, test_h5_2/, ... (for testing)

        The split filtering ensures:
        - Training only uses files from train_* directories (NOT from filenames containing "train")
        - Validation only uses files from val_* directories (NOT from filenames containing "val")
        - Evaluation only uses files from test_* directories (NOT from filenames containing "test")

        CRITICAL: We filter by PARENT DIRECTORY NAME, never by filename. This ensures proper separation
        of training, validation, and test data regardless of how the HDF5 files are named.
        """
        files = []
        subdirs = [d for d in self.data_root.iterdir() if d.is_dir()]

        for subdir in subdirs:
            # Filter by PARENT DIRECTORY NAME (subdir.name), not filename
            subdir_name = subdir.name.lower()

            # Only process subdirectories that match the split
            if self.split == "train":
                if "train" not in subdir_name:
                    continue
            elif self.split == "val":
                if "val" not in subdir_name:
                    continue
            elif self.split == "test":
                if "test" not in subdir_name:
                    continue

            # Only collect files from matching subdirectories
            h5_files = list(subdir.glob("*.h5"))
            files.extend(h5_files)

            # Debug: log which files were found from which directory
            if h5_files:
                logger.info(f"  Found {len(h5_files)} files in '{subdir.name}' (split: {self.split})")

        # Final verification: ensure all files are from the correct parent directories
        verified_files = []
        for f in files:
            parent_dir_name = f.parent.name.lower()
            if self.split == "train" and "train" in parent_dir_name:
                verified_files.append(f)
            elif self.split == "val" and "val" in parent_dir_name:
                verified_files.append(f)
            elif self.split == "test" and "test" in parent_dir_name:
                verified_files.append(f)
            else:
                # This should never happen if logic above is correct
                logger.warning(f"File {f.name} in directory {f.parent.name} doesn't match split '{self.split}'")

        logger.info(f"Found {len(verified_files)} HDF5 files for split '{self.split}' "
                    f"(filtered by PARENT DIRECTORY NAME, not filename)")
        return verified_files

    def _calculate_samples_per_file(self):
        """Calculate number of samples needed for each file based on time windows (not event count)
        
        Uses time-based windows to handle unlimited events per window.
        """
        file_samples = {}

        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, "r") as f:
                    events_group = f["events"]
                    total_events = len(events_group["x"])
                    
                    if total_events == 0:
                        file_samples[h5_file] = 1
                        continue
                    
                    # Calculate samples based on time duration, not event count
                    # Read timestamps to determine time span
                    chunk_size = min(100000, total_events)
                    if total_events <= chunk_size:
                        timestamps = events_group["t"][:]
                        t_min = float(timestamps[0])
                        t_max = float(timestamps[-1])
                    else:
                        # Sample timestamps to estimate range
                        sample_indices = np.linspace(0, total_events - 1, min(1000, total_events), dtype=int)
                        sample_times = events_group["t"][sample_indices]
                        t_min = float(sample_times.min())
                        t_max = float(sample_times.max())
                    
                    total_duration_us = t_max - t_min
                    if total_duration_us <= 0:
                        # All events at same time - use 1 sample
                        samples_needed = 1
                    else:
                        # Calculate number of time windows needed
                        time_window_us = self.time_window_us
                        samples_needed = max(1, int(total_duration_us / time_window_us))
                        # Add some overlap (25%)
                        samples_needed = int(samples_needed * 1.25)

                    # Apply sample limit if configured
                    if self.max_samples_per_file is not None:
                        samples_needed = min(samples_needed, self.max_samples_per_file)

                    file_samples[h5_file] = samples_needed
                    logger.debug(f"  {h5_file.name}: {total_events:,} events, {total_duration_us/1e6:.2f}s duration -> {samples_needed} samples")
            except Exception as e:
                logger.error(f"Error reading {h5_file}: {e}")
                file_samples[h5_file] = 1  # Default to 1 sample if can't read

        return file_samples

    def _create_sample_mapping(self):
        """Create mapping from global sample index to (file_idx, sample_idx)"""
        mapping = {}
        global_idx = 0

        for file_idx, h5_file in enumerate(self.h5_files):
            samples_in_file = self.file_samples[h5_file]

            if self.targeted_training and h5_file in self.annotated_windows:
                # For targeted training, map to annotated window indices
                annotated_indices = self.annotated_windows[h5_file][:samples_in_file]
                for local_sample_idx in range(len(annotated_indices)):
                    # Store the local index (0 to samples_in_file-1) and original index
                    original_sample_idx = annotated_indices[local_sample_idx]
                    mapping[global_idx] = (file_idx, original_sample_idx)
                    global_idx += 1
            else:
                # Standard mapping for non-targeted training
                for sample_idx in range(samples_in_file):
                    mapping[global_idx] = (file_idx, sample_idx)
                    global_idx += 1

        return mapping

    def _load_annotations(self, h5_file_path):
        """Load ALL annotations for an HDF5 file (used internally)"""
        # Check cache first
        if h5_file_path in self._annotation_cache:
            return self._annotation_cache[h5_file_path]

        try:
            # Convert HDF5 filename to annotation filename
            # e.g., train_day_0001_td.h5 -> train_day_0001_bbox.npy
            h5_name = h5_file_path.stem  # Remove .h5
            annotation_name = h5_name.replace("_td", "_bbox") + ".npy"

            possible_paths = []

            # If annotation_dir is provided, use it (for annotations in separate directory)
            # Otherwise, look in the same folder as the h5 file
            if self.annotation_dir:
                # Look in annotation directory structure
                possible_paths.extend([
                    self.annotation_dir / f"eight_class_annotations_{self.split}" / annotation_name,  # Split-specific subdirectory
                    self.annotation_dir / annotation_name,  # Direct fallback
                ])
            else:
                # Look in the same folder as the h5 file
                h5_file_dir = h5_file_path.parent
                possible_paths.append(h5_file_dir / annotation_name)

            annotation_path = None
            for path in possible_paths:
                if path.exists():
                    annotation_path = path
                    break

            if annotation_path:
                annotations = np.load(annotation_path)
                # Cache the loaded annotations
                self._annotation_cache[h5_file_path] = annotations
                return annotations
            else:
                logger.warning(f"Annotation file not found for {h5_name}. Tried paths:")
                for path in possible_paths:
                    logger.warning(f"  - {path}")
                return None

        except Exception as e:
            logger.error(f"Error loading annotations for {h5_file_path}: {e}")
            return None

    def _load_annotations_for_events(self, h5_file_path, event_timestamps):
        """Load annotations that match the specific event timestamps temporally.
        
        Args:
            h5_file_path: Path to the HDF5 file
            event_timestamps: Tensor of event timestamps [N] (not full events tensor to save memory)
        """
        # If annotation_dir is required but not provided, return empty annotations
        # Note: If annotation_dir is None, annotations should be in same folder as h5 files
        if len(event_timestamps) == 0:
            return torch.zeros((0, self.num_classes + 4), dtype=torch.float32, device='cpu')

        try:
            # Load all annotations for the file
            all_annotations = self._load_annotations(h5_file_path)
            if all_annotations is None or len(all_annotations) == 0:
                return torch.zeros((0, 8), dtype=torch.float32, device='cpu')

            # Validate event_timestamps is not empty before calling min/max
            if len(event_timestamps) == 0:
                return torch.zeros((0, 8), dtype=torch.float32, device='cpu')

            # Calculate time window for events
            start_time = float(event_timestamps.min())
            end_time = float(event_timestamps.max())

            # Add larger buffer to account for potential timing differences and capture more annotations
            time_buffer = (end_time - start_time) * 0.2  # 20% buffer (increased from 10% for better annotation capture)
            start_time -= time_buffer
            end_time += time_buffer

            # Filter annotations by time window
            annotation_timestamps = all_annotations['t']
            time_mask = (annotation_timestamps >= start_time) & (annotation_timestamps < end_time)
            filtered_annotations = all_annotations[time_mask]

            if len(filtered_annotations) == 0:
                return torch.zeros((0, 8), dtype=torch.float32, device='cpu')

            # Convert structured array to tensor format: [class_id, x_center, y_center, w, h, conf, track_id, timestamp]
            # Convert from top-left corner format to center format for YOLO compatibility
            targets = torch.zeros((len(filtered_annotations), 8), dtype=torch.float32, device='cpu')

            class_id = torch.from_numpy(filtered_annotations['class_id'].astype(np.float32)).to('cpu')
            targets[:, 0] = class_id  # class_id

            # Convert from [x_left, y_top, w, h] to [x_center, y_center, w, h]
            x_left = torch.from_numpy(filtered_annotations['x'].astype(np.float32)).to('cpu')
            y_top = torch.from_numpy(filtered_annotations['y'].astype(np.float32)).to('cpu')
            w = torch.from_numpy(filtered_annotations['w'].astype(np.float32)).to('cpu')
            h = torch.from_numpy(filtered_annotations['h'].astype(np.float32)).to('cpu')

            x_center = x_left + w / 2  # Convert to center x
            y_center = y_top + h / 2    # Convert to center y

            targets[:, 1] = x_center  # x_center
            targets[:, 2] = y_center  # y_center
            targets[:, 3] = w         # width
            targets[:, 4] = h         # height

            conf = torch.from_numpy(filtered_annotations['class_confidence'].astype(np.float32)).to('cpu')
            track_id = torch.from_numpy(filtered_annotations['track_id'].astype(np.float32)).to('cpu')
            t = torch.from_numpy(filtered_annotations['t'].astype(np.float32)).to('cpu')

            targets[:, 5] = conf  # confidence
            targets[:, 6] = track_id  # track_id
            targets[:, 7] = t  # timestamp

            logger.debug(f"Temporal matching: {len(event_timestamps)} events [{start_time:.0f}-{end_time:.0f}μs] -> {len(filtered_annotations)} annotations")
            return targets

        except Exception as e:
            logger.error(f"Error loading temporal annotations for {h5_file_path}: {e}")
            return torch.zeros((0, 8), dtype=torch.float32, device='cpu')

    def _find_annotated_windows(self):
        """Find time windows that contain annotations by centering on annotation timestamps.

        CRITICAL: This method only operates on files from the current split (train/val/test).
        - self.h5_files is already filtered by split in _find_h5_files() based on parent directory name
        - All operations (class balancing, file diversity) only consider files within the current split
        - Training, validation, and evaluation are completely separated

        Optimized approach:
        1. First, collect ALL unique timestamps from ALL annotation files in the current split (no H5 loading yet)
        2. Select timestamps with class balance and file diversity (based on max_annotations_per_class)
        3. ONLY THEN, for selected timestamps, load corresponding H5 windows
        """
        # STEP 1: Collect all unique timestamps from all annotation files in the CURRENT SPLIT
        # IMPORTANT: self.h5_files is already filtered by split (train/val/test) in _find_h5_files()
        # All subsequent operations (class balancing, file diversity) only work within this split
        # Store both the pairs AND the per-file timestamp maps to avoid duplicate loading
        logger.info(f"STEP 1: Collecting unique timestamps from all annotation files in split '{self.split}'...")
        logger.info(f"Checking {len(self.h5_files)} HDF5 files for annotations...")
        all_timestamps = []  # List of (h5_file, timestamp) tuples - ONLY from current split
        file_timestamp_map = {}  # Pre-build this here to avoid duplicate loading - ONLY from current split
        files_with_annotations = 0
        files_without_annotations = 0

        # Iterate over files that are already filtered by split (train/val/test)
        for h5_file in self.h5_files:
            annotations = self._load_annotations(h5_file)
            if annotations is None or len(annotations) == 0:
                files_without_annotations += 1
                logger.debug(f"  {h5_file.name}: No annotations found")
                continue

            files_with_annotations += 1
            # Get unique annotation timestamps
            annotation_times = annotations['t']
            unique_times = np.unique(annotation_times)
            file_timestamp_map[h5_file] = unique_times  # Store for later use

            # Add (file, timestamp) pairs
            for timestamp in unique_times:
                all_timestamps.append((h5_file, timestamp))
            
            logger.debug(f"  {h5_file.name}: {len(unique_times)} unique timestamps, {len(annotations)} total annotations")

        logger.info(f"Found {len(all_timestamps)} total unique timestamps across {files_with_annotations} files in split '{self.split}'")
        if files_without_annotations > 0:
            logger.warning(f"{files_without_annotations} files had no annotations found")
        
        if len(all_timestamps) == 0:
            error_msg = (
                f"No annotated timestamps found for split '{self.split}'!\n"
                f"Files checked: {len(self.h5_files)}\n"
                f"Files with annotations: {files_with_annotations}\n"
                f"Files without annotations: {files_without_annotations}\n"
                f"Annotation directory: {self.annotation_dir}\n"
                f"Please check:\n"
                f"  1. Annotation files (*_bbox.npy) exist for the HDF5 files\n"
                f"  2. Annotation directory path is correct: {self.annotation_dir}\n"
                f"  3. Annotation files are in the correct format\n"
                f"  4. Annotation files contain valid timestamps"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # STEP 2: Select timestamps based on class balancing if enabled
        # All selection operations (class balancing, file diversity) only work within the current split
        if self.use_class_balanced_sampling:
            # Class-balanced selection: Group by class and ensure balanced representation
            # This balances classes and spreads samples across files - ALL within the current split only
            selected_timestamps = self._class_balanced_selection(all_timestamps, file_timestamp_map)
        else:
            # No class balancing: use all timestamps (or limit by max_samples_per_file if set)
            if self.max_samples_per_file is not None:
                # Limit per file if specified
                logger.info(f"Limiting to {self.max_samples_per_file} samples per file (no class balancing)")
                # This will be handled in _filter_samples_by_annotations
            selected_timestamps = all_timestamps

        logger.info(f"Selected {len(selected_timestamps)} timestamps to evaluate")

        # STEP 3: Group selected timestamps by H5 file
        annotated_windows = {}
        for h5_file in self.h5_files:
            annotated_windows[h5_file] = []

        # file_timestamp_map was already built in STEP 1, no need to reload annotations

        # STEP 4: For each selected timestamp, find its position in the file's unique timestamps
        logger.info("STEP 4: Mapping selected timestamps to sample indices...")
        for h5_file, selected_timestamp in selected_timestamps:
            if h5_file not in file_timestamp_map:
                continue

            # Find the index of this timestamp in the file's unique timestamps
            unique_times = file_timestamp_map[h5_file]
            # Find the index with tolerance for floating point differences
            matches = np.where(np.abs(unique_times - selected_timestamp) < 0.1)[0]

            if len(matches) > 0:
                # Use the first matching index as the sample_idx
                sample_idx = matches[0]
                if sample_idx not in annotated_windows[h5_file]:
                    annotated_windows[h5_file].append(sample_idx)

        logger.info(f"Completed mapping. Windows per file:")
        for h5_file, windows in annotated_windows.items():
            if len(windows) > 0:
                logger.info(f"  {h5_file.name}: {len(windows)} windows")

        return annotated_windows

    def _class_balanced_selection(self, all_timestamps, file_timestamp_map):
        """Select timestamps ensuring balanced annotation counts per class using per-class threads."""
        expected_num_classes = self.num_classes

        cache_key = self._build_balanced_selection_cache_key(expected_num_classes)
        cached_selection = UltraLowMemoryLoader._BALANCED_SELECTION_CACHE.get(cache_key)
        if cached_selection is not None:
            logger.info(f"Using cached class-balanced selection for split '{self.split}'")
            annotation_info = cached_selection.get('annotation_counts')
            if annotation_info:
                logger.debug(f"Cached annotation counts per class: {annotation_info}")
            return cached_selection['selected_timestamps'].copy()

        logger.info(f"Using class-balanced sampling to balance annotation counts per class...")
        logger.info(f"Expected number of classes: {expected_num_classes}")

        estimated_window_us = self.time_window_us
        half_window_us = estimated_window_us / 2

        timestamp_annotation_counts = {}
        for h5_file, timestamp in all_timestamps:
            annotations = self._load_annotations(h5_file)
            if annotations is None or len(annotations) == 0:
                continue

            window_start = timestamp - half_window_us
            window_end = timestamp + half_window_us
            time_mask = (annotations['t'] >= window_start) & (annotations['t'] <= window_end)
            matching_annotations = annotations[time_mask]

            if len(matching_annotations) == 0:
                continue

            class_counts = {}
            for ann in matching_annotations:
                class_id_int = int(ann['class_id'])
                if 0 <= class_id_int < expected_num_classes:
                    class_counts[class_id_int] = class_counts.get(class_id_int, 0) + 1

            if class_counts:
                timestamp_annotation_counts[(h5_file, timestamp)] = class_counts

        logger.info(f"Processed {len(timestamp_annotation_counts)} timestamps with annotations (window {estimated_window_us/1000:.0f}ms)")

        if not timestamp_annotation_counts:
            logger.warning("No annotated timestamps found for balancing. Falling back to timestamp-based balancing.")
            return self._class_balanced_selection_fallback(all_timestamps, file_timestamp_map, expected_num_classes)

        total_annotations_by_class = {i: 0 for i in range(expected_num_classes)}
        for class_counts in timestamp_annotation_counts.values():
            for class_id, count in class_counts.items():
                if class_id in total_annotations_by_class:
                    total_annotations_by_class[class_id] += count

        logger.info(f"Total available annotations per class: {dict(sorted(total_annotations_by_class.items()))}")

        min_class_id = min(total_annotations_by_class.keys(), key=lambda x: total_annotations_by_class[x])
        min_annotations = total_annotations_by_class[min_class_id]
        if min_annotations == 0:
            logger.warning(f"Class {min_class_id} has no annotations available. Falling back to timestamp-based balancing.")
            return self._class_balanced_selection_fallback(all_timestamps, file_timestamp_map, expected_num_classes)

        if self.max_annotations_per_class is not None:
            base_target_annotations = min(min_annotations, self.max_annotations_per_class)
            logger.info(f"Using max_annotations_per_class={self.max_annotations_per_class} (base target {base_target_annotations})")
        else:
            base_target_annotations = min_annotations
            logger.info(f"Balancing to rarest class count: {base_target_annotations} annotations per class")

        target_annotations_per_class_map = {}
        for class_id in range(expected_num_classes):
            available = total_annotations_by_class[class_id]
            target = min(base_target_annotations, available) if available > 0 else 0
            target_annotations_per_class_map[class_id] = target

        logger.info(f"Per-class targets: {dict(sorted(target_annotations_per_class_map.items()))}")

        timestamps_by_file_and_class = {}
        for h5_file in self.h5_files:
            timestamps_by_file_and_class[h5_file] = {i: [] for i in range(expected_num_classes)}

        for ts_key, class_counts_dict in timestamp_annotation_counts.items():
            h5_file, _ = ts_key
            if h5_file not in timestamps_by_file_and_class:
                timestamps_by_file_and_class[h5_file] = {i: [] for i in range(expected_num_classes)}
            for class_id in range(expected_num_classes):
                if class_counts_dict.get(class_id, 0) > 0:
                    timestamps_by_file_and_class[h5_file][class_id].append((ts_key, class_counts_dict))

        all_files = sorted(timestamps_by_file_and_class.keys(), key=lambda f: str(f))

        class_candidates = {}
        for class_id in range(expected_num_classes):
            per_file_lists = [timestamps_by_file_and_class[file_path][class_id]
                              for file_path in all_files
                              if timestamps_by_file_and_class[file_path][class_id]]
            interleaved = []
            if per_file_lists:
                max_len = max(len(lst) for lst in per_file_lists)
                for i in range(max_len):
                    for lst in per_file_lists:
                        if i < len(lst):
                            interleaved.append(lst[i])
            class_candidates[class_id] = interleaved

        selection_lock = Lock()
        selected_timestamp_set = set()
        selection_order = []
        current_annotation_counts = {i: 0 for i in range(expected_num_classes)}
        tolerance = 1.05

        def class_worker(class_id: int):
            target_limit = target_annotations_per_class_map.get(class_id, 0)
            if target_limit <= 0:
                logger.debug(f"  Class {class_id}: no target annotations (skipping)")
                return

            candidates = class_candidates.get(class_id, [])
            if not candidates:
                logger.debug(f"  Class {class_id}: no candidate timestamps found")
                return

            num_candidates = len(candidates)
            if num_candidates == 0:
                logger.debug(f"  Class {class_id}: no candidate timestamps found")
                return
            max_attempts = max(num_candidates * 5, num_candidates)
            attempts = 0

            while attempts < max_attempts:
                with selection_lock:
                    if current_annotation_counts[class_id] >= target_limit:
                        break

                ts_key, class_counts_dict = candidates[attempts % num_candidates]
                attempts += 1

                if class_counts_dict.get(class_id, 0) == 0:
                    continue

                with selection_lock:
                    if current_annotation_counts[class_id] >= target_limit:
                        break

                    if ts_key in selected_timestamp_set:
                        continue

                    exceeds_limits = False
                    for cid, count in class_counts_dict.items():
                        if cid >= expected_num_classes:
                            continue
                        limit = target_annotations_per_class_map.get(cid, 0)
                        if limit <= 0:
                            continue
                        projected = current_annotation_counts[cid] + count
                        if cid == class_id:
                            if projected > limit:
                                exceeds_limits = True
                                break
                        else:
                            if projected > limit * tolerance:
                                exceeds_limits = True
                                break

                    if exceeds_limits:
                        continue

                    selected_timestamp_set.add(ts_key)
                    selection_order.append(ts_key)
                    for cid, count in class_counts_dict.items():
                        if cid < expected_num_classes:
                            current_annotation_counts[cid] += count

            with selection_lock:
                logger.debug(f"  Class {class_id}: collected {current_annotation_counts[class_id]} annotations (target {target_limit})")

        threads = []
        for class_id in range(expected_num_classes):
            thread = Thread(target=class_worker, args=(class_id,), daemon=True)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if not selection_order:
            logger.warning("Multi-thread class selection produced no timestamps. Falling back to timestamp-based balancing.")
            return self._class_balanced_selection_fallback(all_timestamps, file_timestamp_map, expected_num_classes)

        unique_selected = []
        seen = set()
        for ts_key in selection_order:
            if ts_key not in seen:
                unique_selected.append(ts_key)
                seen.add(ts_key)

        final_annotation_counts = {i: 0 for i in range(expected_num_classes)}
        for ts_key in unique_selected:
            class_counts_dict = timestamp_annotation_counts.get(ts_key, {})
            for cid, count in class_counts_dict.items():
                if cid < expected_num_classes:
                    final_annotation_counts[cid] += count

        for class_id in range(expected_num_classes):
            limit = target_annotations_per_class_map.get(class_id, 0)
            if limit <= 0:
                continue
            if final_annotation_counts[class_id] <= limit:
                continue
            idx = len(unique_selected) - 1
            while idx >= 0 and final_annotation_counts[class_id] > limit:
                ts_key = unique_selected[idx]
                class_counts_dict = timestamp_annotation_counts.get(ts_key, {})
                if class_counts_dict.get(class_id, 0) > 0:
                    unique_selected.pop(idx)
                    for cid, count in class_counts_dict.items():
                        if cid < expected_num_classes:
                            final_annotation_counts[cid] -= count
                    if ts_key in seen:
                        seen.remove(ts_key)
                    idx = len(unique_selected) - 1
                    continue
                idx -= 1

        if not unique_selected:
            logger.warning("After trimming per-class limits no timestamps remain. Falling back to timestamp-based balancing.")
            return self._class_balanced_selection_fallback(all_timestamps, file_timestamp_map, expected_num_classes)

        logger.info(f"Selected {len(unique_selected)} unique timestamps after multi-thread selection")
        logger.info(f"Final annotation counts per class: {dict(sorted(final_annotation_counts.items()))}")

        file_usage_counts = {}
        for ts_key in unique_selected:
            h5_file = ts_key[0]
            file_usage_counts[h5_file] = file_usage_counts.get(h5_file, 0) + 1

        if file_usage_counts:
            file_counts_sorted = sorted(file_usage_counts.items(), key=lambda x: x[1], reverse=True)
            max_per_file = file_counts_sorted[0][1]
            min_per_file = file_counts_sorted[-1][1]
            if max_per_file > 0:
                file_diversity_ratio = min_per_file / max_per_file
                logger.debug(f"File diversity ratio (min/max samples per file): {file_diversity_ratio:.3f} (1.0 = perfect diversity)")

        if final_annotation_counts:
            max_count = max(final_annotation_counts.values())
            min_count = min(final_annotation_counts.values())
            if max_count > 0:
                balance_ratio = min_count / max_count
                logger.debug(f"Class balance ratio (min/max): {balance_ratio:.3f} (1.0 = perfect balance)")

        found_classes = {cid for cid, count in final_annotation_counts.items() if count > 0}
        missing_classes = set(range(expected_num_classes)) - found_classes
        if missing_classes:
            logger.warning(f"Some classes have no annotations in selected samples: {sorted(missing_classes)}")

        UltraLowMemoryLoader._BALANCED_SELECTION_CACHE[cache_key] = {
            'selected_timestamps': unique_selected.copy(),
            'annotation_counts': dict(sorted(final_annotation_counts.items())),
            'num_files': len(file_usage_counts)
        }
        logger.debug(f"Cached class-balanced selection for split '{self.split}' (key length {len(cache_key)})")

        return unique_selected

    def _build_balanced_selection_cache_key(self, expected_num_classes: int) -> str:
        """Build a cache key for class-balanced selection based on configuration and files."""
        file_list = tuple(sorted(str(f.resolve()) for f in self.h5_files))
        key_components = (
            f"split={self.split}",
            f"classes={expected_num_classes}",
            f"max_annotations_per_class={self.max_annotations_per_class}",
            f"min_samples_per_class={self.min_samples_per_class}",
            f"use_class_balanced_sampling={self.use_class_balanced_sampling}",
            f"num_classes={self.num_classes}",
            f"targeted_training={self.targeted_training}",
            f"max_samples_per_file={self.max_samples_per_file}",
            f"files={file_list}"
        )
        return "|".join(str(component) for component in key_components)

    def _class_balanced_selection_fallback(self, all_timestamps, file_timestamp_map, expected_num_classes):
        """Fallback to original timestamp-based balancing if annotation-based fails."""
        logger.info("Falling back to timestamp-based class balancing...")
        # Group timestamps by class (original method)
        class_timestamps = {}
        for h5_file, timestamp in all_timestamps:
            annotations = self._load_annotations(h5_file)
            if annotations is None or len(annotations) == 0:
                continue
            time_mask = np.abs(annotations['t'] - timestamp) < 0.1
            matching_annotations = annotations[time_mask]
            if len(matching_annotations) > 0:
                class_ids = np.unique(matching_annotations['class_id'])
                for class_id in class_ids:
                    class_id_int = int(class_id)
                    if 0 <= class_id_int < expected_num_classes:
                        if class_id_int not in class_timestamps:
                            class_timestamps[class_id_int] = []
                        class_timestamps[class_id_int].append((h5_file, timestamp))

        # Simple balanced selection
        selected_timestamps = []
        used_timestamps = set()
        # Fallback: select all timestamps (no limit when using fallback)

        for class_id, timestamps in class_timestamps.items():
            num_to_select = len(timestamps)
            # Ensure seed is within valid range [0, 2**32 - 1]
            seed_value = (42 + int(class_id)) % (2**32)
            np.random.seed(seed_value)
            selected_indices = np.random.choice(len(timestamps), num_to_select, replace=False)
            for idx in selected_indices:
                ts_pair = timestamps[idx]
                if ts_pair not in used_timestamps:
                    selected_timestamps.append(ts_pair)
                    used_timestamps.add(ts_pair)

        return selected_timestamps

    def _filter_samples_by_annotations(self):
        """Filter samples to only include those with annotations when targeted_training=True"""
        file_samples = {}

        for h5_file in self.h5_files:
            if h5_file in self.annotated_windows:
                # Only count samples that have annotations
                annotated_indices = self.annotated_windows[h5_file]
                num_annotated_samples = len(annotated_indices)

                # Apply max_samples_per_file limit if configured (only when not using class balancing globally)
                # When class balancing is enabled globally, selection already happened in _find_annotated_windows
                if self.max_samples_per_file is not None and not self.use_class_balanced_sampling:
                    # Simple limit: take first N samples (no class balancing)
                    num_annotated_samples = min(num_annotated_samples, self.max_samples_per_file)
                    # Update annotated_windows to only include selected samples
                    self.annotated_windows[h5_file] = annotated_indices[:num_annotated_samples]
                elif self.use_class_balanced_sampling and num_annotated_samples > 0:
                    # Class balancing already handled globally - just ensure we have the selected samples
                    # The annotated_windows already contains the selected samples from _class_balanced_selection
                    pass

                file_samples[h5_file] = num_annotated_samples
                logger.info(f"  {h5_file.name}: {num_annotated_samples} annotated samples")
            else:
                file_samples[h5_file] = 0
                logger.warning(f"  {h5_file.name}: No annotations found")

        return file_samples

    def _apply_class_balanced_limit_per_file(self, h5_file, annotated_indices, max_samples):
        """Apply class-balanced sampling limit per file."""
        if not self.use_class_balanced_sampling or len(annotated_indices) == 0:
            return min(len(annotated_indices), max_samples)

        # Load annotations to get class information for each timestamp index
        annotations = self._load_annotations(h5_file)
        if annotations is None or len(annotations) == 0:
            return min(len(annotated_indices), max_samples)

        # Get unique timestamps from the file
        annotation_times = annotations['t']
        unique_times = np.unique(annotation_times)

        # Group annotated indices by class
        class_indices = {}  # {class_id: [sample_idx, ...]}

        for sample_idx in annotated_indices:
            if sample_idx < len(unique_times):
                timestamp = unique_times[sample_idx]
                # Find annotations at this timestamp
                time_mask = np.abs(annotation_times - timestamp) < 0.1
                matching = annotations[time_mask]

                if len(matching) > 0:
                    # Get unique class IDs at this timestamp
                    class_ids = np.unique(matching['class_id'])

                    # Determine expected number of classes
                    expected_num_classes = self.num_classes

                    for class_id in class_ids:
                        class_id_int = int(class_id)
                        # Validate class ID is within expected range
                        if 0 <= class_id_int < expected_num_classes:
                            if class_id_int not in class_indices:
                                class_indices[class_id_int] = []
                            if sample_idx not in class_indices[class_id_int]:
                                class_indices[class_id_int].append(sample_idx)
                        else:
                            # Warn about invalid class IDs but continue
                            logger.warning(f"Found class ID {class_id_int} outside expected range [0, {expected_num_classes-1}]. Skipping.")

        # If no classes found, return original limit
        if len(class_indices) == 0:
            return min(len(annotated_indices), max_samples)

        # Select samples ensuring class balance
        selected_indices = []
        used_indices = set()

        # First, ensure minimum samples per class
        # Double-check that class_indices is not empty (already checked, but be safe)
        if len(class_indices) == 0:
            return min(len(annotated_indices), max_samples)

        samples_per_class = max(1, max_samples // len(class_indices))
        min_samples = min(self.min_samples_per_class, samples_per_class)

        for class_id in sorted(class_indices.keys()):
            class_samples = class_indices[class_id]
            num_to_select = min(min_samples, len(class_samples))

            if num_to_select > 0:
                # Use a hash-based seed that's reproducible (hash of file path + class_id)
                # Convert Path to string and use hash for reproducibility
                file_hash = hash(str(h5_file)) % (2**31)  # Limit to 32-bit signed int
                # Ensure seed is within valid range [0, 2**32 - 1]
                seed_value = (42 + file_hash + int(class_id)) % (2**32)
                np.random.seed(seed_value)  # Reproducible
                selected = np.random.choice(class_samples, num_to_select, replace=False)
                for idx in selected:
                    if idx not in used_indices:
                        selected_indices.append(idx)
                        used_indices.add(idx)

        # Fill remaining slots with balanced sampling
        remaining = max_samples - len(selected_indices)
        if remaining > 0 and len(class_indices) > 0:
            samples_per_class = max(1, remaining // len(class_indices))
            remainder = remaining % len(class_indices)

            for i, class_id in enumerate(sorted(class_indices.keys())):
                if len(selected_indices) >= max_samples:
                    break

                available = [idx for idx in class_indices[class_id] if idx not in used_indices]
                if len(available) > 0:
                    num_to_select = samples_per_class + (1 if i < remainder else 0)
                    num_to_select = min(num_to_select, len(available), max_samples - len(selected_indices))

                    if num_to_select > 0:
                        # Use a hash-based seed that's reproducible
                        file_hash = hash(str(h5_file)) % (2**31)  # Limit to 32-bit signed int
                        # Ensure seed is within valid range [0, 2**32 - 1]
                        seed_value = (42 + file_hash + int(class_id) + i) % (2**32)
                        np.random.seed(seed_value)
                        selected = np.random.choice(available, num_to_select, replace=False)
                        for idx in selected:
                            if idx not in used_indices:
                                selected_indices.append(idx)
                                used_indices.add(idx)
                                if len(selected_indices) >= max_samples:
                                    break

        # Update annotated_windows to only include selected indices
        selected_indices.sort()
        self.annotated_windows[h5_file] = selected_indices

        # Print class distribution for this file
        if len(selected_indices) > 0:
            class_counts = {}
            expected_num_classes = self.num_classes

            for sample_idx in selected_indices:
                if sample_idx < len(unique_times):
                    timestamp = unique_times[sample_idx]
                    time_mask = np.abs(annotation_times - timestamp) < 0.1
                    matching = annotations[time_mask]
                    if len(matching) > 0:
                        class_ids = np.unique(matching['class_id'])
                        for class_id in class_ids:
                            class_id_int = int(class_id)
                            # Only count valid class IDs
                            if 0 <= class_id_int < expected_num_classes:
                                class_counts[class_id_int] = class_counts.get(class_id_int, 0) + 1
            logger.info(f"    {h5_file.name} class distribution: {dict(sorted(class_counts.items()))}")

        return len(selected_indices)


    def _preload_all_windows(self):
        """
        PRE-LOAD ALL SAMPLES INTO RAM FOR MAXIMUM SPEED!

        This loads all event windows and annotations into the sample cache during initialization.
        Trades RAM for speed - eliminates ALL I/O during training!

        Expected RAM usage: ~5-10GB for 2500 samples
        Expected speedup: 5-10x (batches from 8s -> 1-2s)
        """

        start_time = time.time()
        print_interval = max(1, self.total_samples // 20)  # Print progress every 5%

        for idx in range(self.total_samples):
            try:
                # Load sample (will be cached automatically by __getitem__)
                _ = self.__getitem__(idx)

                # Print progress
                if (idx + 1) % print_interval == 0 or idx == 0:
                    elapsed = time.time() - start_time
                    progress_pct = ((idx + 1) / self.total_samples) * 100
                    samples_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
                    eta_seconds = (self.total_samples - idx - 1) / samples_per_sec if samples_per_sec > 0 else 0

                    logger.info(f"  Progress: {idx + 1}/{self.total_samples} samples ({progress_pct:.1f}%) | "
                                f"Speed: {samples_per_sec:.1f} samples/s | "
                                f"ETA: {eta_seconds/60:.1f} min")

            except Exception as e:
                logger.warning(f"Failed to preload sample {idx}: {e}")
                continue

        elapsed = time.time() - start_time
        logger.info(f"\nPreloading completed in {elapsed/60:.2f} minutes")
        logger.info(f"Average speed: {self.total_samples/elapsed:.1f} samples/second")

        # Calculate memory usage estimate (events removed from cache to save memory)
        if self._sample_cache:
            cache_size_mb = sum(
                item['frames'].element_size() * item['frames'].nelement() +
                item['targets'].element_size() * item['targets'].nelement() +
                item['event_timestamps'].element_size() * item['event_timestamps'].nelement()
                for item in self._sample_cache.values()
            ) / (1024 * 1024)
            logger.info(f"Estimated cache size: {cache_size_mb:.1f} MB ({cache_size_mb/1024:.2f} GB)")

    def __len__(self):
        return self.total_samples

    def _find_event_window_fast(self, h5_file, target_timestamp, use_cache_key=None):
        """
        OPTIMIZED: Use binary search on HDF5 timestamps without loading entire array.
        Returns (start_idx, end_idx) for the event window.
        """
        # Check precomputed cache first
        if use_cache_key and use_cache_key in self._event_indices_cache:
            return self._event_indices_cache[use_cache_key]

        try:
            # Open HDF5 with SWMR mode for reduced contention
            with h5py.File(h5_file, "r", swmr=True, libver='latest') as f:
                events_group = f["events"]
                total_events = len(events_group["t"])

                if total_events == 0:
                    return (0, 0)

                # OPTIMIZATION 1: Use binary search on chunked reads instead of loading all timestamps
                # Read timestamp array in chunks to find the target window efficiently
                chunk_size = 100000  # Read 100k timestamps at a time
                # Use time window instead of event count limit - handle unlimited events
                half_window_us = self.time_window_us / 2  # Half the time window in microseconds

                # Binary search to find approximate location
                left, right = 0, total_events - 1
                target_idx = None

                # Quick binary search with chunked reads
                while right - left > chunk_size:
                    mid = (left + right) // 2
                    mid_time = events_group["t"][mid]

                    if mid_time < target_timestamp:
                        left = mid
                    else:
                        right = mid

                # Now read a larger chunk around the found region
                search_start = max(0, left - chunk_size // 2)
                search_end = min(total_events, right + chunk_size // 2)
                local_timestamps = events_group["t"][search_start:search_end]

                # Find closest index within this chunk
                local_distances = np.abs(local_timestamps - target_timestamp)
                local_closest = np.argmin(local_distances)
                closest_idx = search_start + local_closest

                # Calculate window bounds based on TIME, not event count
                # Load all events within the time window (can be 3-4M events)
                target_time = float(events_group["t"][closest_idx])
                window_start_time = target_time - half_window_us
                window_end_time = target_time + half_window_us

                # Find all events within the time window using binary search
                # Use chunked binary search to handle large event counts efficiently
                # Find start index
                left, right = 0, closest_idx
                # Use chunked reads for binary search to avoid loading all timestamps
                while right - left > chunk_size:
                    mid = (left + right) // 2
                    mid_time = float(events_group["t"][mid])
                    if mid_time < window_start_time:
                        left = mid
                    else:
                        right = mid
                
                # Final binary search on smaller range
                while right - left > 1:
                    mid = (left + right) // 2
                    mid_time = float(events_group["t"][mid])
                    if mid_time < window_start_time:
                        left = mid + 1
                    else:
                        right = mid
                start_idx = left

                # Find end index
                left, right = closest_idx, total_events - 1
                # Use chunked reads for binary search
                while right - left > chunk_size:
                    mid = (left + right) // 2
                    mid_time = float(events_group["t"][mid])
                    if mid_time <= window_end_time:
                        left = mid
                    else:
                        right = mid
                
                # Final binary search on smaller range
                while right - left > 1:
                    mid = (left + right) // 2
                    mid_time = float(events_group["t"][mid])
                    if mid_time <= window_end_time:
                        left = mid + 1
                    else:
                        right = mid
                end_idx = right

                # Ensure we have at least some events
                if end_idx <= start_idx:
                    # Fallback: use a small window around the target
                    start_idx = max(0, closest_idx - 10000)
                    end_idx = min(total_events, closest_idx + 10000)

                num_events_in_window = end_idx - start_idx
                if num_events_in_window > 0:
                    logger.debug(f"Time window [{window_start_time:.0f}-{window_end_time:.0f}μs] contains {num_events_in_window:,} events (unlimited)")

                result = (start_idx, end_idx)

                # Cache the result
                if use_cache_key:
                    self._event_indices_cache[use_cache_key] = result

                return result

        except Exception as e:
            logger.error(f"Error in _find_event_window_fast for {h5_file}: {e}")
            # Fallback: return a reasonable window (no limit on event count)
            return (0, min(1000000, total_events))  # 1M events as safe fallback

    def _load_events_batch(self, h5_file, start_idx, end_idx):
        """
        OPTIMIZED: Load all event arrays (x,y,t,p) in one go using fancy indexing.
        Returns stacked tensor [N, 4].
        NOTE: For large event counts (3-4M+), use _events_to_frames_streaming instead.
        
        WARNING: This method loads all events into memory at once. For large event counts,
        this can cause CUDA OOM errors. This method should NOT be used in __getitem__.
        Use _events_to_frames_streaming instead, which streams events in chunks.
        """
        try:
            # Open with SWMR mode
            with h5py.File(h5_file, "r", swmr=True, libver='latest') as f:
                events_group = f["events"]

                if start_idx >= end_idx:
                    events = torch.zeros((0, 4), dtype=torch.float32)
                    if self.force_cpu:
                        events = events.cpu()
                    return events

                # OPTIMIZATION 2: Create index array once
                indices = np.arange(start_idx, end_idx)

                # OPTIMIZATION 3: Read all arrays - HDF5 will optimize this internally
                x = events_group["x"][indices]
                y = events_group["y"][indices]
                t = events_group["t"][indices]
                p = events_group["p"][indices]

                # OPTIMIZATION 4: Stack numpy arrays first, then convert to tensor (faster)
                events_np = np.stack([x, y, t, p], axis=1).astype(np.float32)
                events = torch.from_numpy(events_np)

                if self.force_cpu:
                    events = events.cpu()

                return events

        except Exception as e:
            logger.error(f"Error loading events from {h5_file}: {e}")
            events = torch.zeros((0, 4), dtype=torch.float32)
            if self.force_cpu:
                events = events.cpu()
            return events

    def _events_to_frames_streaming(self, h5_file, start_idx, end_idx):
        """
        Stream events directly from HDF5 in chunks and accumulate frames incrementally.
        This avoids loading all events (3-4M+) into memory at once.
        Returns frames [T, H, W] and event_timestamps [N].
        """
        # Determine output geometry
        if not hasattr(self, 'time_steps'):
            raise ValueError("UltraLowMemoryLoader.time_steps is not set. "
                             "Ensure the dataset was constructed with a valid time_steps.")
        T = int(self.time_steps)
        H = getattr(self, 'image_height', 720)
        W = getattr(self, 'image_width', 1280)

        # Create empty frames
        # ALWAYS use CPU in DataLoader workers (forked processes cannot use CUDA)
        # The training loop will move tensors to GPU later
        dtype = torch.float32
        device = 'cpu'  # Always CPU in DataLoader to avoid CUDA fork issues
        frames = torch.zeros((T, H, W), dtype=dtype, device=device)

        if start_idx >= end_idx:
            # Return empty timestamps (min/max for annotation matching)
            event_timestamps = torch.zeros(0, dtype=torch.float32, device=device)
            return frames, event_timestamps

        num_events = end_idx - start_idx
        chunk_size = 100000  # Stream 100K events at a time to minimize memory usage (reduced from 500K)
        
        # Initialize min/max timestamps for annotation matching
        t_min_val = None
        t_max_val = None
        
        try:
            with h5py.File(h5_file, "r", swmr=True, libver='latest') as f:
                events_group = f["events"]
                
                # First pass: Get time range for binning AND annotation matching
                # We only need min/max for annotations, not all timestamps
                # CRITICAL: Use .copy() to break HDF5 dataset reference
                if num_events > chunk_size * 2:
                    # Read first chunk timestamps
                    first_chunk_t = events_group["t"][start_idx:min(start_idx + chunk_size, end_idx)].copy()
                    # Read last chunk timestamps
                    last_chunk_t = events_group["t"][max(start_idx, end_idx - chunk_size):end_idx].copy()
                    t_min = float(min(np.min(first_chunk_t), np.min(last_chunk_t)))
                    t_max = float(max(np.max(first_chunk_t), np.max(last_chunk_t)))
                    del first_chunk_t, last_chunk_t
                else:
                    # Small enough to read all timestamps at once
                    t_all = events_group["t"][start_idx:end_idx].copy()
                    t_min = float(np.min(t_all))
                    t_max = float(np.max(t_all))
                    del t_all
                
                # Store min/max for annotation matching (we don't need all timestamps)
                t_min_val = t_min
                t_max_val = t_max

                # Guard against degenerate timestamps
                if (t_max - t_min) <= 0:
                    bin_edges = None
                else:
                    bin_edges = torch.linspace(t_min, t_max, T + 1, device=device)

                # Second pass: Stream events in chunks and accumulate frames
                if num_events > chunk_size:
                    num_chunks = (num_events + chunk_size - 1) // chunk_size
                    logger.debug(f"Streaming {num_events:,} events from HDF5 in {num_chunks} chunks to save memory")
                    
                    for chunk_idx in range(num_chunks):
                        chunk_start = start_idx + chunk_idx * chunk_size
                        chunk_end = min(start_idx + (chunk_idx + 1) * chunk_size, end_idx)
                        
                        # Read chunk directly from HDF5 (only this chunk in memory)
                        # CRITICAL: Use .copy() to break HDF5 dataset reference and create independent numpy arrays
                        x_chunk = events_group["x"][chunk_start:chunk_end].copy()
                        y_chunk = events_group["y"][chunk_start:chunk_end].copy()
                        t_chunk = events_group["t"][chunk_start:chunk_end].copy()
                        p_chunk = events_group["p"][chunk_start:chunk_end].copy()
                        
                        # Convert to tensors - use .clone() to ensure tensor owns its memory
                        # This breaks the numpy array reference completely
                        x_t = torch.from_numpy(x_chunk.astype(np.float32, copy=False)).clone().to(device)
                        y_t = torch.from_numpy(y_chunk.astype(np.float32, copy=False)).clone().to(device)
                        t_t = torch.from_numpy(t_chunk.astype(np.float32, copy=False)).clone().to(device)
                        p_t = torch.from_numpy(p_chunk.astype(np.float32, copy=False)).clone().to(device)
                        
                        # Delete numpy arrays immediately (critical for memory)
                        # Now safe to delete since tensors have their own memory via .clone()
                        del x_chunk, y_chunk, t_chunk, p_chunk
                        import gc
                        gc.collect()  # Force GC after each chunk
                        
                        # Explicitly clear CUDA cache if CUDA is available (safety measure)
                        # Even though we use CPU device, this ensures no accidental CUDA allocations persist
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Bucketize timestamps
                        if bin_edges is not None:
                            b_chunk = torch.bucketize(t_t.contiguous(), bin_edges) - 1
                            b_chunk = b_chunk.clamp_(0, T - 1)
                        else:
                            b_chunk = torch.full((len(t_t),), T - 1, device=device, dtype=torch.long)
                        
                        # Integer pixel coords
                        xi_chunk = x_t.long().clamp_(0, W - 1)
                        yi_chunk = y_t.long().clamp_(0, H - 1)
                        
                        # Signed polarity
                        if torch.any(p_t < 0):
                            val_chunk = p_t.to(dtype)
                        else:
                            val_chunk = (2.0 * p_t.to(dtype) - 1.0)
                        
                        # Accumulate into frames
                        frames.index_put_((b_chunk, yi_chunk, xi_chunk), val_chunk, accumulate=True)
                        
                        # Delete chunk tensors immediately
                        del x_t, y_t, t_t, p_t, b_chunk, xi_chunk, yi_chunk, val_chunk
                        
                        # Force garbage collection after each chunk to free memory immediately
                        import gc
                        gc.collect()
                        
                        # Explicitly clear CUDA cache if CUDA is available (safety measure)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Small number of events - read all at once but still process efficiently
                    # CRITICAL: Use .copy() to break HDF5 dataset reference
                    x = events_group["x"][start_idx:end_idx].copy()
                    y = events_group["y"][start_idx:end_idx].copy()
                    t = events_group["t"][start_idx:end_idx].copy()
                    p = events_group["p"][start_idx:end_idx].copy()
                    
                    # Convert to tensors - use .clone() to ensure tensor owns its memory
                    x_t = torch.from_numpy(x.astype(np.float32, copy=False)).clone().to(device)
                    y_t = torch.from_numpy(y.astype(np.float32, copy=False)).clone().to(device)
                    t_t = torch.from_numpy(t.astype(np.float32, copy=False)).clone().to(device)
                    p_t = torch.from_numpy(p.astype(np.float32, copy=False)).clone().to(device)
                    
                    # Delete numpy arrays immediately (now safe since tensors have own memory)
                    del x, y, t, p
                    import gc
                    gc.collect()
                    
                    # Bucketize
                    if bin_edges is not None:
                        b = torch.bucketize(t_t.contiguous(), bin_edges) - 1
                        b = b.clamp_(0, T - 1)
                    else:
                        b = torch.full((len(t_t),), T - 1, device=device, dtype=torch.long)
                    
                    # Integer pixel coords
                    xi = x_t.long().clamp_(0, W - 1)
                    yi = y_t.long().clamp_(0, H - 1)
                    
                    # Signed polarity
                    if torch.any(p_t < 0):
                        val = p_t.to(dtype)
                    else:
                        val = (2.0 * p_t.to(dtype) - 1.0)
                    
                    # Accumulate
                    frames.index_put_((b, yi, xi), val, accumulate=True)
                    
                    # Delete immediately
                    del x_t, y_t, t_t, p_t, b, xi, yi, val
                    import gc
                    gc.collect()
                    
                    # Explicitly clear CUDA cache if CUDA is available (safety measure)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # HDF5 file handle is automatically closed by 'with' statement
                # But explicitly clear any remaining references after processing
                del events_group
                import gc
                gc.collect()
                
                # Explicitly clear CUDA cache after processing all events
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error streaming events from {h5_file}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty timestamps on error
            event_timestamps = torch.zeros(0, dtype=torch.float32, device=device)
            return frames, event_timestamps

        # Create event_timestamps tensor with min/max for annotation matching
        # We only need min/max, not all timestamps (saves huge memory for 3-4M events)
        # Create a small tensor with [min, max] for _load_annotations_for_events
        if t_min_val is not None and t_max_val is not None:
            event_timestamps = torch.tensor([t_min_val, t_max_val], dtype=torch.float32, device=device)
        else:
            event_timestamps = torch.zeros(0, dtype=torch.float32, device=device)

        # SCALE AND CLIP FOR SNN STABILITY
        scale = getattr(self, "event_frame_scale", 20.0)
        clip = getattr(self, "event_frame_clip", 4.0)

        if scale is not None and scale > 0:
            frames = frames / float(scale)

        frames = frames.clamp_(-float(clip), float(clip))

        return frames, event_timestamps

    def _events_to_frames(self, events: torch.Tensor) -> torch.Tensor:
        """Rasterize raw events [N,4]=[x,y,t,p] into signed spike frames [T,H,W].
        ON events contribute +1, OFF events contribute -1. Binning is uniform in time.
        Processes events in chunks and immediately deletes event data to save memory.
        If self.time_steps is unavailable, defaults to 8. Image size defaults to 720x1280
        unless self.image_height/self.image_width are defined.
        
        WARNING: This method loads all events into memory first. For large event counts (3-4M+),
        use _events_to_frames_streaming instead to avoid CUDA OOM errors.
        This method is kept for backward compatibility but should not be used in __getitem__.
        """
        # Determine output geometry
        if not hasattr(self, 'time_steps'):
            raise ValueError("UltraLowMemoryLoader.time_steps is not set. "
                             "Ensure the dataset was constructed with a valid time_steps.")
        T = int(self.time_steps)
        H = getattr(self, 'image_height', 720)
        W = getattr(self, 'image_width', 1280)

        # Create empty frames on CPU by default; move to events.device if needed later
        device = events.device
        dtype = torch.float32
        frames = torch.zeros((T, H, W), dtype=dtype, device=device)

        # Handle empty window
        if events.numel() == 0:
            return frames

        num_events = events.shape[0]
        chunk_size = 1000000  # Process 1M events at a time to avoid 32-bit limits and memory issues
        
        # Extract time range first (needed for binning) - use a small chunk to get min/max
        if num_events > chunk_size:
            # Sample first and last chunks to get time range
            first_chunk_t = events[:min(chunk_size, num_events), 2]
            last_chunk_t = events[max(0, num_events - chunk_size):, 2]
            t_min = min(first_chunk_t.min().item(), last_chunk_t.min().item())
            t_max = max(first_chunk_t.max().item(), last_chunk_t.max().item())
            del first_chunk_t, last_chunk_t
        else:
            t = events[:, 2]
            t_min = t.min().item()
            t_max = t.max().item()
            del t

        # Guard against degenerate timestamps
        if (t_max - t_min) <= 0:
            # Put all events into the last bin to avoid division-by-zero issues
            bin_edges = None
        else:
            # Build bin edges for bucketizing
            bin_edges = torch.linspace(t_min, t_max, T + 1, device=device)

        # Process events in chunks to accumulate frames incrementally
        # This allows us to delete event chunks immediately after processing
        if num_events > chunk_size:
            num_chunks = (num_events + chunk_size - 1) // chunk_size
            logger.debug(f"Processing {num_events:,} events in {num_chunks} chunks to save memory and avoid 32-bit index limits")
            
            for chunk_start in range(0, num_events, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_events)
                
                # Extract chunk data (creates new tensors, not views)
                chunk_events = events[chunk_start:chunk_end].clone()  # Clone to break reference
                x_chunk = chunk_events[:, 0]
                y_chunk = chunk_events[:, 1]
                t_chunk = chunk_events[:, 2]
                p_chunk = chunk_events[:, 3]
                
                # Delete the chunk immediately after extracting columns
                del chunk_events
                
                # Bucketize timestamps for this chunk
                if bin_edges is not None:
                    b_chunk = torch.bucketize(t_chunk.contiguous(), bin_edges) - 1
                    b_chunk = b_chunk.clamp_(0, T - 1)
                else:
                    b_chunk = torch.full((len(t_chunk),), T - 1, device=device, dtype=torch.long)
                
                # Integer pixel coords with bounds check
                xi_chunk = x_chunk.long().clamp_(0, W - 1)
                yi_chunk = y_chunk.long().clamp_(0, H - 1)
                
                # Signed polarity: assume p∈{-1,1}; if {0,1}, map 0->-1 via (2*p-1)
                if torch.any(p_chunk < 0):
                    val_chunk = p_chunk.to(dtype)
                else:
                    val_chunk = (2.0 * p_chunk.to(dtype) - 1.0)
                
                # Accumulate into frames
                frames.index_put_((b_chunk, yi_chunk, xi_chunk), val_chunk, accumulate=True)
                
                # Delete chunk tensors immediately
                del x_chunk, y_chunk, t_chunk, p_chunk, b_chunk, xi_chunk, yi_chunk, val_chunk
                
                # Force garbage collection after each chunk
                import gc
                gc.collect()
                
                # Explicitly clear CUDA cache if CUDA is available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Small number of events - process all at once but still delete immediately
            x = events[:, 0].clone()
            y = events[:, 1].clone()
            t = events[:, 2].clone()
            p = events[:, 3].clone()
            
            # Bucketize timestamps
            if bin_edges is not None:
                b = torch.bucketize(t.contiguous(), bin_edges) - 1
                b = b.clamp_(0, T - 1)
            else:
                b = torch.full((num_events,), T - 1, device=device, dtype=torch.long)
            
            # Integer pixel coords with bounds check
            xi = x.long().clamp_(0, W - 1)
            yi = y.long().clamp_(0, H - 1)
            
            # Signed polarity
            if torch.any(p < 0):
                val = p.to(dtype)
            else:
                val = (2.0 * p.to(dtype) - 1.0)
            
            # Accumulate into frames
            frames.index_put_((b, yi, xi), val, accumulate=True)
            
            # Delete immediately
            del x, y, t, p, b, xi, yi, val
            import gc
            gc.collect()
            
            # Explicitly clear CUDA cache if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # SCALE AND CLIP FOR SNN STABILITY:
        # Large raw counts can saturate MultiSpike4 / Integer LIF neurons.
        # Normalize by a configurable scale factor and clamp to a small range.
        # If attributes are not set on the dataset, fall back to safe defaults.
        scale = getattr(self, "event_frame_scale", 20.0)
        clip = getattr(self, "event_frame_clip", 4.0)

        if scale is not None and scale > 0:
            frames = frames / float(scale)

        # Clamp to a bounded range (compatible with MultiSpike4's useful range)
        frames = frames.clamp_(-float(clip), float(clip))
        
        # Final cleanup - ensure all event data is freed
        import gc
        gc.collect()
        
        # Explicitly clear CUDA cache if CUDA is available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return frames

    def __getitem__(self, idx):# Use the new mapping system
        file_idx, sample_idx = self.sample_mapping[idx]
        h5_file = self.h5_files[file_idx]
        samples_in_file = self.file_samples[h5_file]
        cache_key = (file_idx, sample_idx)
        if self.cache_samples and self._sample_cache is not None and cache_key in self._sample_cache:
            cached = self._sample_cache[cache_key]
            return {
                "frames": cached["frames"].clone(),  # Pre-computed frames - USED BY MODEL
                "targets": cached["targets"].clone(),
                "event_timestamps": cached["event_timestamps"].clone(),
                "filename": cached["filename"],
                "sample_idx": sample_idx,
            }

        # Debug printing
        if self.debug_sample_loading:
            if self.targeted_training and h5_file in self.annotated_windows:
                annotated_indices = self.annotated_windows[h5_file][:samples_in_file]
                local_idx = (
                    annotated_indices.index(sample_idx) + 1
                    if sample_idx in annotated_indices
                    else sample_idx + 1
                )
                logger.debug(f"Streaming from: {h5_file.name} (sample {local_idx}/{samples_in_file})")
            else:
                logger.debug(f"Streaming from: {h5_file.name} (sample {sample_idx + 1}/{samples_in_file})")

        start_idx = 0
        end_idx = 0
        target_ann_time = None

        try:
            # OPTIMIZED PATH: targeted training + annotations
            if self.targeted_training and h5_file in self.annotated_windows and self.annotation_dir:
                annotations = self._load_annotations(h5_file)
                if annotations is not None and len(annotations) > 0:
                    annotation_times = annotations["t"]
                    unique_times = np.unique(annotation_times)
                    if sample_idx < len(unique_times):
                        target_ann_time = unique_times[sample_idx]

                # Binary search for window (requires a target time)
                if target_ann_time is not None:
                    start_idx, end_idx = self._find_event_window_fast(
                        h5_file, target_ann_time, use_cache_key=cache_key
                    )
                else:
                    # Fallback: no valid target time → sliding window
                    with h5py.File(h5_file, "r", swmr=True, libver="latest") as f:
                        events_group = f["events"]
                        total_events = len(events_group["x"])
                        if total_events == 0:
                            start_idx = 0
                            end_idx = 0
                        else:
                            indices = self._get_event_indices(events_group, sample_idx, total_events)
                            start_idx = indices[0] if len(indices) > 0 else 0
                            end_idx = indices[-1] + 1 if len(indices) > 0 else 0

            else:
                # No annotations → sliding window method
                with h5py.File(h5_file, "r", swmr=True, libver="latest") as f:
                    events_group = f["events"]
                    total_events = len(events_group["x"])

                    if total_events == 0:
                        start_idx = 0
                        end_idx = 0
                    else:
                        indices = self._get_event_indices(events_group, sample_idx, total_events)
                        start_idx = indices[0] if len(indices) > 0 else 0
                        end_idx = indices[-1] + 1 if len(indices) > 0 else 0

        except Exception as e:
            logger.error(f"Error determining event window for {h5_file}: {e}")
            traceback.print_exc()
            start_idx = 0
            end_idx = 0

        # STREAMING: Process events directly from HDF5 in chunks and accumulate frames
        # This avoids loading all events (3-4M+) into memory at once
        # The function streams events in 100K chunks, processes them, and deletes immediately
        # CRITICAL: Always use _events_to_frames_streaming, never _events_to_frames or _load_events_batch
        # to prevent CUDA OOM errors
        if self.debug_sample_loading:
            logger.debug(f"Using streaming method for {h5_file.name} (events {start_idx}-{end_idx})")
        frames, event_timestamps = self._events_to_frames_streaming(h5_file, start_idx, end_idx)
        
        # Load corresponding annotations with temporal matching (uses event_timestamps)
        targets = self._load_annotations_for_events(h5_file, event_timestamps)
        
        # Force garbage collection to free any remaining memory
        import gc
        gc.collect()
        
        # Explicitly clear CUDA cache after loading sample to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self.force_cpu:
            frames = frames.cpu()

        # Cache if needed (frames only - events removed to save memory)
        if self.cache_samples and self._sample_cache is not None:
            self._sample_cache[cache_key] = {
                "frames": frames.clone(),
                "targets": targets.clone(),
                "event_timestamps": event_timestamps.clone(),
                "filename": h5_file.name,
            }

        # Return only frames (model input) - raw events removed to save memory
        # Frames are [T,H,W] which is what the model expects
        # Raw events are no longer returned since they're not used by the model
        return {
            "frames": frames,            # [T,H,W] signed spike frames - USED BY MODEL
            "targets": targets,
            "event_timestamps": event_timestamps,
            "filename": h5_file.name,
            "sample_idx": sample_idx,
        }

    def _get_event_indices(self, events_group, sample_idx, total_events):
        """Get event indices using time-based sliding window approach (for non-targeted training)
        
        Uses time windows instead of event count limits to handle unlimited events per window.
        """
        # Use time-based windows instead of event count limits
        # Calculate time window size in microseconds
        time_window_us = self.time_window_us
        
        # Get timestamps to calculate time-based windows
        if total_events == 0:
            return np.arange(0)
        
        # Read timestamps in chunks to avoid loading all at once
        chunk_size = 100000
        if total_events <= chunk_size:
            timestamps = events_group["t"][:]
            t_min = float(timestamps[0])
            t_max = float(timestamps[-1])
        else:
            # Sample timestamps to estimate range
            sample_indices = np.linspace(0, total_events - 1, min(1000, total_events), dtype=int)
            sample_times = events_group["t"][sample_indices]
            t_min = float(sample_times.min())
            t_max = float(sample_times.max())
        
        total_duration = t_max - t_min
        if total_duration <= 0:
            # All events at same time - return all
            return np.arange(total_events)
        
        # Calculate number of time windows
        num_windows = max(1, int(total_duration / time_window_us))
        
        # Calculate which window this sample belongs to
        window_idx = sample_idx % num_windows
        window_start_time = t_min + window_idx * time_window_us
        window_end_time = min(t_max, window_start_time + time_window_us)
        
        # Binary search to find events in this time window
        # Find start index
        left, right = 0, total_events - 1
        while right - left > 1:
            mid = (left + right) // 2
            mid_time = float(events_group["t"][mid])
            if mid_time < window_start_time:
                left = mid + 1
            else:
                right = mid
        start_idx = left
        
        # Find end index
        left, right = start_idx, total_events - 1
        while right - left > 1:
            mid = (left + right) // 2
            mid_time = float(events_group["t"][mid])
            if mid_time <= window_end_time:
                left = mid + 1
            else:
                right = mid
        end_idx = right
        
        # Ensure we have at least some events
        if end_idx <= start_idx:
            # Fallback: use a small window
            start_idx = max(0, sample_idx * 10000)
            end_idx = min(total_events, start_idx + 10000)
        
        return np.arange(start_idx, end_idx)

def custom_collate_fn(batch, force_cpu=False):
    """Custom collate function to handle frames and targets.
    
    NOTE: Raw events are no longer included in the batch to save memory.
    Only frames [T,H,W] are returned, which is what the model uses.
    """
    targets = [item["targets"] for item in batch]
    event_timestamps = [item["event_timestamps"] for item in batch]
    filenames = [item["filename"] for item in batch]
    sample_indices = [item["sample_idx"] for item in batch]
    frames_list = [item.get("frames") for item in batch]

    # CRITICAL: Force all input tensors to CPU ONLY if force_cpu is True
    # Worker processes might create tensors on CUDA even with pin_memory=False
    if force_cpu:
        targets = [t.cpu() if t.is_cuda else t for t in targets]
        event_timestamps = [ts.cpu() if ts.is_cuda else ts for ts in event_timestamps]

    # Handle targets with different sizes (some samples may have 0 annotations)
    if len(targets) == 0:
        device = 'cpu' if force_cpu else None
        targets_tensor = torch.zeros((0, 8), dtype=torch.float32, device=device)  # Updated to 8 to include timestamp
    else:
        # Find the maximum number of annotations in any sample
        max_annotations = max(target.shape[0] for target in targets)

        # Pad all targets to the same size
        padded_targets = []
        for target in targets:
            # Ensure target is on CPU if force_cpu
            if force_cpu:
                target = target.cpu()
            if target.shape[0] < max_annotations:
                # Pad with zeros - explicitly use CPU device if force_cpu
                device = 'cpu' if force_cpu else target.device
                padding = torch.zeros((max_annotations - target.shape[0], target.shape[1]), dtype=target.dtype, device=device)
                padded_target = torch.cat([target, padding], dim=0)
                if force_cpu:
                    padded_target = padded_target.cpu()
            else:
                padded_target = target.cpu() if force_cpu else target
            padded_targets.append(padded_target)

        targets_tensor = torch.stack(padded_targets, dim=0)
        if force_cpu:
            targets_tensor = targets_tensor.cpu()

    # Stack frames: each is [T,H,W]; ensure CPU if force_cpu
    if len(frames_list) == 0 or frames_list[0] is None:
        frames_tensor = None
    else:
        frames_tensor = None
        frames_list_proc = []
        for f in frames_list:
            if f is None:
                frames_tensor = None
                break
            frames_list_proc.append(f.cpu() if force_cpu else f)
        else:
            frames_tensor = torch.stack(frames_list_proc, dim=0)  # [B,T,H,W]

    # Pad event timestamps to same length for the batch
    # Get max length from event_timestamps (they correspond to the original events)
    if len(event_timestamps) == 0:
        max_length = 0
    else:
        max_length = max(ts.shape[0] for ts in event_timestamps)
    
    padded_timestamps = []
    for ts in event_timestamps:
        # Ensure ts is on CPU if force_cpu
        if force_cpu:
            ts = ts.cpu()
        if ts.shape[0] < max_length:
            # Pad with -1 (invalid timestamp marker) - explicitly use CPU device if force_cpu
            device = 'cpu' if force_cpu else ts.device
            padding = torch.zeros(max_length - ts.shape[0], dtype=ts.dtype, device=device) - 1
            padded_ts = torch.cat([ts, padding], dim=0)
            if force_cpu:
                padded_ts = padded_ts.cpu()
        else:
            padded_ts = ts.cpu() if force_cpu else ts
        padded_timestamps.append(padded_ts)

    event_timestamps_tensor = torch.stack(padded_timestamps, dim=0)
    if force_cpu:
        event_timestamps_tensor = event_timestamps_tensor.cpu()

    return {
        "frames": frames_tensor,                 # [B, T, H, W] - model input
        "targets": targets_tensor,
        "event_timestamps": event_timestamps_tensor,  # Event timestamps for temporal matching
        "filenames": filenames,
        "sample_indices": sample_indices
    }

from typing import Optional

def create_ultra_low_memory_dataloader(data_root: str, split: str = "train",
                                      batch_size: int = 4, max_events_per_sample: int = 10000,
                                      num_workers: int = 4, shuffle: bool = False, annotation_dir: str = None,
                                      max_samples_per_file: int = None, targeted_training: bool = True,
                                      force_cpu: bool = False,
                                      num_classes: int = 3, drop_last: bool = True,
                                      use_class_balanced_sampling: bool = False, min_samples_per_class: int = 1,
                                      max_annotations_per_class: int = None,
                                      cache_samples: bool = False, preload_all_samples: bool = False,
                                      debug_sample_loading: bool = False,
                                      time_steps: Optional[int] = None,
                                      image_height: int = 720, image_width: int = 1280, time_window_us: int = 100000,
                                      config: Optional[Any] = None):
    logger.info("=== CREATING ULTRA LOW MEMORY DATALOADER ===")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Split: {split}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max events per sample: {max_events_per_sample}")
    logger.info(f"Max samples per file: {max_samples_per_file}")
    logger.info(f"Max annotations per class: {max_annotations_per_class}")
    logger.info(f"Targeted training: {targeted_training}")
    logger.info(f"Num workers: {num_workers}")
    logger.info(f"Annotation dir: {annotation_dir}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Use class-balanced sampling: {use_class_balanced_sampling}")
    logger.info(f"Min samples per class: {min_samples_per_class}")
    logger.info(f"Cache samples: {cache_samples}")
    logger.info(f"Time steps (T): {time_steps}")
    logger.info(f"Frame size (H×W): {image_height}×{image_width}")

    if time_steps is None:
        raise ValueError("time_steps must be provided (no hardcoded default). Pass config.get_time_steps().")

    dataset = UltraLowMemoryLoader(
        data_root=data_root,
        split=split,
        max_events_per_sample=max_events_per_sample,
        annotation_dir=annotation_dir,
        max_samples_per_file=max_samples_per_file,
        targeted_training=targeted_training,
        force_cpu=force_cpu,  # Pass force_cpu to dataset
        num_classes=num_classes,  # Pass number of classes
        use_class_balanced_sampling=use_class_balanced_sampling,  # Pass class balancing flag
        min_samples_per_class=min_samples_per_class,  # Pass minimum samples per class
        max_annotations_per_class=max_annotations_per_class,  # Pass annotation limit per class
        cache_samples=cache_samples,
        preload_all_samples=preload_all_samples,  # Pass preloading flag
        debug_sample_loading=debug_sample_loading,
        time_steps=time_steps,
        image_height=image_height,
        image_width=image_width,
        time_window_us=time_window_us
    )

    logger.info(f"Dataset created with {len(dataset)} samples")
    logger.info(f"Force CPU: {force_cpu}")

    # Create a collate function wrapper that captures force_cpu
    def collate_wrapper(batch):
        return custom_collate_fn(batch, force_cpu=force_cpu)

    # Get DataLoader parameters from config if available, otherwise use defaults
    if config is not None:
        prefetch_factor_config = config.get('data_processing.prefetch_factor', 4)
        persistent_workers_config = config.get('data_processing.persistent_workers', True)
        pin_memory_config = config.get('data_processing.pin_memory', True)
    else:
        prefetch_factor_config = 4
        persistent_workers_config = True
        pin_memory_config = True

    # When num_workers=0, prefetch_factor must be None
    prefetch_factor_val = prefetch_factor_config if num_workers > 0 else None
    # Use config value for persistent_workers (can be disabled for small datasets)
    persistent_workers_val = persistent_workers_config
    # Use config value for pin_memory (can be disabled to avoid CUDA errors)
    pin_memory_val = pin_memory_config

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory_val,  # Use config value (can be disabled to avoid CUDA errors)
        prefetch_factor=prefetch_factor_val,  # Prefetch batches for faster loading (only if using workers)
        persistent_workers=persistent_workers_val,  # Keep workers alive between epochs
        drop_last=drop_last,  # Use drop_last parameter (False for validation to keep incomplete batches)
        collate_fn=collate_wrapper  # Use wrapper with force_cpu
    )

    logger.info(f"Ultra Low Memory DataLoader created with {len(data_loader)} batches")
    logger.info("=== ULTRA LOW MEMORY DATALOADER CREATION COMPLETE ===")

    return data_loader
