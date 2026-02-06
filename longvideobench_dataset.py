# Copyright 2024 LongVideoBench Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import random
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import av
import numpy as np


@dataclass
class LongVideoBenchSample:
    """A single sample from LongVideoBench dataset."""
    video_id: str
    question: str
    options: List[str]
    answer: str
    category: str
    subcategory: str
    video_path: str
    start_time: float
    end_time: float
    duration: float
    difficulty: str
    reasoning_type: str


class LongVideoBenchDataset(Dataset):
    """LongVideoBench dataset for video question answering."""
    
    def __init__(
        self,
        data_path: str,
        video_dir: str,
        split: str = "test",
        max_num_frames: int = 8,
        video_fps: int = 1,
        sample_type: str = "uniform",
        transform=None,
    ):
        """Initialize LongVideoBench dataset.
        
        Args:
            data_path: Path to the annotation file
            video_dir: Directory containing video files
            split: Dataset split (train/val/test)
            max_num_frames: Maximum number of frames to extract
            video_fps: FPS for frame extraction
            sample_type: Sampling strategy (uniform/random)
            transform: Image transform
        """
        self.data_path = data_path
        self.video_dir = video_dir
        self.split = split
        self.max_num_frames = max_num_frames
        self.video_fps = video_fps
        self.sample_type = sample_type
        self.transform = transform
        
        # Load annotations
        self.samples = self._load_annotations()
        
    def _load_annotations(self) -> List[LongVideoBenchSample]:
        """Load annotations from JSON file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            # Filter by split if specified
            if 'split' in item and item['split'] != self.split:
                continue
                
            sample = LongVideoBenchSample(
                video_id=item['video_id'],
                question=item['question'],
                options=item['options'],
                answer=item['answer'],
                category=item.get('category', ''),
                subcategory=item.get('subcategory', ''),
                video_path=item['video_path'],
                start_time=item.get('start_time', 0),
                end_time=item.get('end_time', -1),
                duration=item.get('duration', 0),
                difficulty=item.get('difficulty', 'medium'),
                reasoning_type=item.get('reasoning_type', '')
            )
            samples.append(sample)
            
        return samples
    
    def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video file."""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            # Get video duration
            duration = float(container.duration / av.time_base)
            
            # Calculate frame indices
            if self.sample_type == "uniform":
                frame_indices = np.linspace(
                    0, duration - 1, self.max_num_frames, dtype=int
                )
            else:  # random
                frame_indices = sorted(
                    random.sample(
                        range(int(duration)), 
                        min(self.max_num_frames, int(duration))
                    )
                )
            
            frames = []
            for frame_idx in frame_indices:
                container.seek(int(frame_idx * av.time_base))
                for frame in container.decode(video_stream):
                    if frame.pts >= frame_idx * video_stream.time_base:
                        img_array = frame.to_ndarray(format='rgb24')
                        frames.append(img_array)
                        break
            
            container.close()
            return frames
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        
        # Extract video frames
        video_path = os.path.join(self.video_dir, sample.video_path)
        frames = self._extract_video_frames(video_path)
        
        # Convert frames to PIL Images and apply transforms
        if self.transform:
            frames = [self.transform(Image.fromarray(frame)) for frame in frames]
        else:
            frames = [Image.fromarray(frame) for frame in frames]
        
        # Stack frames if needed
        if len(frames) > 0:
            frames = torch.stack(frames)
        else:
            # Create dummy frames if extraction failed
            frames = torch.zeros(self.max_num_frames, 3, 224, 224)
        
        return {
            'video_id': sample.video_id,
            'frames': frames,
            'question': sample.question,
            'options': sample.options,
            'answer': sample.answer,
            'category': sample.category,
            'subcategory': sample.subcategory,
            'difficulty': sample.difficulty,
            'reasoning_type': sample.reasoning_type
        }


class LongVideoBenchCollator:
    """Collator for LongVideoBench dataset."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch of samples."""
        # Extract video frames
        frames = [item['frames'] for item in batch]
        
        # Pad frames to same length
        max_frames = max(len(f) for f in frames)
        padded_frames = []
        for f in frames:
            if len(f) < max_frames:
                padding = torch.zeros(max_frames - len(f), *f.shape[1:])
                f = torch.cat([f, padding], dim=0)
            padded_frames.append(f)
        
        frames = torch.stack(padded_frames)
        
        # Prepare text inputs
        questions = [item['question'] for item in batch]
        options_list = [item['options'] for item in batch]
        
        # Format question with options
        formatted_questions = []
        for q, options in zip(questions, options_list):
            formatted_q = q + "\n"
            for i, opt in enumerate(options):
                formatted_q += f"{chr(65+i)}. {opt}\n"
            formatted_questions.append(formatted_q)
        
        # Tokenize questions
        text_inputs = self.tokenizer(
            formatted_questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'frames': frames,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'video_ids': [item['video_id'] for item in batch],
            'answers': [item['answer'] for item in batch],
            'categories': [item['category'] for item in batch],
            'difficulties': [item['difficulty'] for item in batch]
        }


def create_longvideobench_dataset(
    data_path: str,
    video_dir: str,
    split: str = "test",
    **kwargs
) -> LongVideoBenchDataset:
    """Create LongVideoBench dataset."""
    return LongVideoBenchDataset(
        data_path=data_path,
        video_dir=video_dir,
        split=split,
        **kwargs
    )


def get_dataset_statistics(dataset: LongVideoBenchDataset) -> Dict[str, Any]:
    """Get dataset statistics."""
    categories = {}
    difficulties = {}
    reasoning_types = {}
    
    for sample in dataset.samples:
        # Count categories
        cat = sample.category
        categories[cat] = categories.get(cat, 0) + 1
        
        # Count difficulties
        diff = sample.difficulty
        difficulties[diff] = difficulties.get(diff, 0) + 1
        
        # Count reasoning types
        rtype = sample.reasoning_type
        reasoning_types[rtype] = reasoning_types.get(rtype, 0) + 1
    
    return {
        'total_samples': len(dataset.samples),
        'categories': categories,
        'difficulties': difficulties,
        'reasoning_types': reasoning_types
    }