#!/usr/bin/env python3
"""
Vidaio Local Validation Script

This script implements the exact scoring formulas used by Vidaio subnet for:
1. Upscaling tasks (PIE-APP + VMAF + Content Length)
2. Compression tasks (Compression Rate + VMAF)
3. Performance multipliers (Bonus/Penalty system)

Usage:
    python local_validation.py --task upscaling --reference video.mp4 --processed processed.mp4
    python local_validation.py --task compression --reference video.mp4 --processed compressed.mp4 --threshold 85
    python local_validation.py --calculate-multipliers --history-folder ./performance_history
"""

import os
import sys
import math
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import subprocess
from datetime import datetime

# Add the project root to Python path
sys.path.append('/workspace/vidaio-subnet')

try:
    from services.scoring.pieapp_metric import calculate_pieapp_score
    from services.scoring.vmaf_metric import calculate_vmaf, convert_mp4_to_y4m
    from services.scoring.scoring_function import calculate_compression_score
except ImportError as e:
    print(f"Warning: Could not import Vidaio scoring modules: {e}")
    print("Some features may not work. Make sure you're in the vidaio-subnet directory.")

class VidaioScorer:
    """
    Implements the exact Vidaio scoring formulas for local validation.
    """
    
    def __init__(self):
        # Constants from vidaio_subnet_core/validating/managing/miner_manager.py
        self.VMAF_THRESHOLD = 0.5  # From CONFIG.score.vmaf_threshold
        self.PIEAPP_THRESHOLD = 1.0  # From CONFIG.score.pieapp_threshold
        self.VMAF_SAMPLE_COUNT = 10  # From CONFIG.score.vmaf_sample_count
        self.PIEAPP_SAMPLE_COUNT = 4  # From CONFIG.score.pieapp_sample_count
        
        # Upscaling constants
        self.QUALITY_WEIGHT = 0.5
        self.LENGTH_WEIGHT = 0.5
        self.MIN_CONTENT_LENGTH = 5.0
        self.TARGET_CONTENT_LENGTH = 30.0
        
        # Compression constants
        self.COMPRESSION_RATE_WEIGHT = 0.8  # w_c
        self.COMPRESSION_VMAF_WEIGHT = 0.2  # w_vmaf
        self.SOFT_THRESHOLD_MARGIN = 5.0
        
        # Performance multiplier constants
        self.UPSCALING_BONUS_THRESHOLD = 0.32
        self.PENALTY_F_THRESHOLD = 0.07
        self.PENALTY_Q_THRESHOLD = 0.5
        self.BONUS_MAX = 0.15
        self.PENALTY_F_MAX = 0.20
        self.PENALTY_Q_MAX = 0.25
        
        # Compression multiplier constants
        self.COMPRESSION_BONUS_THRESHOLD = 0.74
        self.COMPRESSION_PENALTY_F_THRESHOLD = 0.4
        self.COMPRESSION_VMAF_MARGIN = 0.0
        self.COMPRESSION_BONUS_MAX = 0.15
        self.COMPRESSION_PENALTY_F_MAX = 0.20
        self.COMPRESSION_PENALTY_VMAF_MAX = 0.30

    def calculate_vmaf_score(self, reference_path: str, processed_path: str) -> float:
        """
        Calculate VMAF score using harmonic mean of frame scores.
        
        Args:
            reference_path: Path to reference video
            processed_path: Path to processed video
            
        Returns:
            VMAF score (0-100)
        """
        try:
            # Get video info
            ref_cap = cv2.VideoCapture(reference_path)
            if not ref_cap.isOpened():
                raise ValueError(f"Cannot open reference video: {reference_path}")
            
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ref_cap.release()
            
            if ref_total_frames < self.VMAF_SAMPLE_COUNT:
                raise ValueError(f"Reference video has only {ref_total_frames} frames, need at least {self.VMAF_SAMPLE_COUNT}")
            
            # Sample random frames for VMAF calculation
            random_frames = sorted(np.random.choice(ref_total_frames, 
                                                  min(self.VMAF_SAMPLE_COUNT, ref_total_frames), 
                                                  replace=False))
            
            # Convert to Y4M format for VMAF
            ref_y4m_path = convert_mp4_to_y4m(reference_path, random_frames)
            
            try:
                vmaf_score = calculate_vmaf(ref_y4m_path, processed_path, random_frames, neg_model=False)
                return vmaf_score if vmaf_score is not None else 0.0
            finally:
                # Cleanup
                if os.path.exists(ref_y4m_path):
                    os.unlink(ref_y4m_path)
                    
        except Exception as e:
            print(f"Error calculating VMAF: {e}")
            return 0.0

    def calculate_pieapp_score(self, reference_path: str, processed_path: str) -> float:
        """
        Calculate PIE-APP score using the exact Vidaio implementation.
        
        Args:
            reference_path: Path to reference video
            processed_path: Path to processed video
            
        Returns:
            PIE-APP score (0-2, lower is better)
        """
        try:
            ref_cap = cv2.VideoCapture(reference_path)
            dist_cap = cv2.VideoCapture(processed_path)
            
            if not ref_cap.isOpened() or not dist_cap.isOpened():
                raise ValueError("Cannot open video files")
            
            ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dist_total_frames = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if ref_total_frames != dist_total_frames:
                raise ValueError(f"Frame count mismatch: ref={ref_total_frames}, dist={dist_total_frames}")
            
            if ref_total_frames < self.PIEAPP_SAMPLE_COUNT:
                raise ValueError(f"Video has only {ref_total_frames} frames, need at least {self.PIEAPP_SAMPLE_COUNT}")
            
            # Sample consecutive frames
            sample_size = min(self.PIEAPP_SAMPLE_COUNT, ref_total_frames)
            max_start_frame = ref_total_frames - sample_size
            start_frame = 0 if max_start_frame <= 0 else np.random.randint(0, max_start_frame)
            
            # Extract reference frames
            ref_frames = []
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = ref_cap.read()
                if not ret:
                    break
                ref_frames.append(frame)
            
            # Extract processed frames
            dist_frames = []
            dist_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(sample_size):
                ret, frame = dist_cap.read()
                if not ret:
                    break
                dist_frames.append(frame)
            
            ref_cap.release()
            dist_cap.release()
            
            # Calculate PIE-APP score using Vidaio's method
            pieapp_score = self._calculate_pieapp_score_on_samples(ref_frames, dist_frames)
            return pieapp_score
            
        except Exception as e:
            print(f"Error calculating PIE-APP: {e}")
            return 2.0  # Max penalty on error

    def _calculate_pieapp_score_on_samples(self, ref_frames: List[np.ndarray], dist_frames: List[np.ndarray]) -> float:
        """
        Calculate PIE-APP score on sampled frames (exact Vidaio implementation).
        """
        if not ref_frames or not dist_frames:
            return 2.0
        
        # Create frame provider classes that mimic cv2.VideoCapture
        class FrameProvider:
            def __init__(self, frames):
                self.frames = frames
                self.current_frame = 0
                self.frame_count = len(frames)
            
            def read(self):
                if self.current_frame < self.frame_count:
                    frame = self.frames[self.current_frame]
                    self.current_frame += 1
                    return True, frame
                return False, None
            
            def get(self, prop_id):
                if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                    return self.frame_count
                return 0
            
            def set(self, prop_id, value):
                if prop_id == cv2.CAP_PROP_POS_FRAMES:
                    self.current_frame = int(value) if value < self.frame_count else self.frame_count
                    return True
                return False
            
            def release(self):
                pass
            
            def isOpened(self):
                return self.frame_count > 0
        
        ref_provider = FrameProvider(ref_frames)
        dist_provider = FrameProvider(dist_frames)
        
        try:
            score = calculate_pieapp_score(ref_provider, dist_provider, frame_interval=1)
            return score
        except Exception as e:
            print(f"Error in PIE-APP calculation: {e}")
            return 2.0

    def calculate_content_length_score(self, content_length: float) -> float:
        """
        Calculate content length score using Vidaio's logarithmic formula.
        
        Formula: S_L = log(1 + content_length) / log(1 + 320)
        
        Args:
            content_length: Video duration in seconds
            
        Returns:
            Length score (0-1)
        """
        return math.log(1 + content_length) / math.log(1 + 320)

    def calculate_upscaling_quality_score(self, vmaf_score: float, pieapp_score: float) -> float:
        """
        Calculate quality score for upscaling tasks.
        
        Args:
            vmaf_score: VMAF score (0-100)
            pieapp_score: PIE-APP score (0-2, lower is better)
            
        Returns:
            Quality score (0-1)
        """
        # Check VMAF threshold first
        if vmaf_score / 100 < self.VMAF_THRESHOLD:
            return 0.0
        
        # Apply PIE-APP transformation (exact Vidaio formula)
        return self._calculate_final_score_from_pieapp(pieapp_score)

    def _calculate_final_score_from_pieapp(self, pieapp_score: float) -> float:
        """
        Calculate final score from PIE-APP using Vidaio's exact transformation.
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        sigmoid_normalized_score = sigmoid(pieapp_score)
        
        original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
        original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
        
        original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
        
        scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
        
        return scaled_value

    def calculate_upscaling_final_score(self, s_q: float, s_l: float) -> float:
        """
        Calculate final upscaling score using exponential transformation.
        
        Formula: S_F = 0.1 × e^(6.979 × (S_pre - 0.5))
        where S_pre = 0.5 × S_Q + 0.5 × S_L
        
        Args:
            s_q: Quality score (0-1)
            s_l: Length score (0-1)
            
        Returns:
            Final score (exponential scale)
        """
        s_pre = self.QUALITY_WEIGHT * s_q + self.LENGTH_WEIGHT * s_l
        s_f = 0.1 * math.exp(6.979 * (s_pre - 0.5))
        return s_f

    def calculate_compression_score(self, vmaf_score: float, compression_rate: float, vmaf_threshold: float) -> Tuple[float, float, float, str]:
        """
        Calculate compression score using Vidaio's exact formula.
        
        Args:
            vmaf_score: VMAF quality score (0-100)
            compression_rate: Size ratio compressed/original (0-1, lower is better)
            vmaf_threshold: Required VMAF threshold
            
        Returns:
            Tuple of (final_score, compression_component, quality_component, reason)
        """
        return calculate_compression_score(
            vmaf_score=vmaf_score,
            compression_rate=compression_rate,
            vmaf_threshold=vmaf_threshold,
            compression_weight=self.COMPRESSION_RATE_WEIGHT,
            quality_weight=self.COMPRESSION_VMAF_WEIGHT,
            soft_threshold_margin=self.SOFT_THRESHOLD_MARGIN
        )

    def calculate_performance_multipliers(self, performance_history: List[Dict]) -> Dict[str, float]:
        """
        Calculate performance multipliers based on rolling 10-round history.
        
        Args:
            performance_history: List of performance records with keys:
                - s_f: Final score
                - s_q: Quality score (for upscaling)
                - vmaf_score: VMAF score
                - vmaf_threshold: VMAF threshold
                - task_type: "upscaling" or "compression"
        
        Returns:
            Dictionary with multiplier values
        """
        if len(performance_history) == 0:
            return {
                "bonus_multiplier": 1.0,
                "penalty_f_multiplier": 1.0,
                "penalty_q_multiplier": 1.0,
                "total_multiplier": 1.0
            }
        
        # Use last 10 rounds
        recent_records = performance_history[-10:]
        
        # Count bonuses and penalties
        bonus_count = 0
        penalty_f_count = 0
        penalty_q_count = 0
        
        for record in recent_records:
            s_f = record.get("s_f", 0.0)
            s_q = record.get("s_q", 0.0)
            vmaf_score = record.get("vmaf_score", 0.0)
            vmaf_threshold = record.get("vmaf_threshold", 0.0)
            task_type = record.get("task_type", "upscaling")
            
            if task_type == "upscaling":
                # Upscaling thresholds
                if s_f > self.UPSCALING_BONUS_THRESHOLD:
                    bonus_count += 1
                if s_f < self.PENALTY_F_THRESHOLD:
                    penalty_f_count += 1
                if s_q < self.PENALTY_Q_THRESHOLD:
                    penalty_q_count += 1
            else:
                # Compression thresholds
                if s_f > self.COMPRESSION_BONUS_THRESHOLD:
                    bonus_count += 1
                if s_f < self.COMPRESSION_PENALTY_F_THRESHOLD:
                    penalty_f_count += 1
                if vmaf_score < vmaf_threshold + self.COMPRESSION_VMAF_MARGIN:
                    penalty_q_count += 1
        
        # Calculate multipliers
        bonus_multiplier = 1.0 + (bonus_count / 10) * self.BONUS_MAX
        penalty_f_multiplier = 1.0 - (penalty_f_count / 10) * self.PENALTY_F_MAX
        penalty_q_multiplier = 1.0 - (penalty_q_count / 10) * self.PENALTY_Q_MAX
        
        total_multiplier = bonus_multiplier * penalty_f_multiplier * penalty_q_multiplier
        
        return {
            "bonus_multiplier": bonus_multiplier,
            "penalty_f_multiplier": penalty_f_multiplier,
            "penalty_q_multiplier": penalty_q_multiplier,
            "total_multiplier": total_multiplier,
            "bonus_count": bonus_count,
            "penalty_f_count": penalty_f_count,
            "penalty_q_count": penalty_q_count
        }

    def score_upscaling_task(self, reference_path: str, processed_path: str, content_length: float) -> Dict:
        """
        Score an upscaling task using Vidaio's exact formulas.
        
        Args:
            reference_path: Path to reference video
            processed_path: Path to processed video
            content_length: Video duration in seconds
            
        Returns:
            Dictionary with all scores and metrics
        """
        print("🎬 Calculating VMAF score...")
        vmaf_score = self.calculate_vmaf_score(reference_path, processed_path)
        
        print("🎨 Calculating PIE-APP score...")
        pieapp_score = self.calculate_pieapp_score(reference_path, processed_path)
        
        print("📏 Calculating content length score...")
        s_l = self.calculate_content_length_score(content_length)
        
        print("🔍 Calculating quality score...")
        s_q = self.calculate_upscaling_quality_score(vmaf_score, pieapp_score)
        
        print("⚡ Calculating final score...")
        s_f = self.calculate_upscaling_final_score(s_q, s_l)
        
        return {
            "task_type": "upscaling",
            "vmaf_score": vmaf_score,
            "pieapp_score": pieapp_score,
            "s_q": s_q,
            "s_l": s_l,
            "s_f": s_f,
            "content_length": content_length,
            "vmaf_threshold_passed": vmaf_score / 100 >= self.VMAF_THRESHOLD,
            "pieapp_threshold_passed": pieapp_score <= self.PIEAPP_THRESHOLD
        }

    def score_compression_task(self, reference_path: str, processed_path: str, vmaf_threshold: float) -> Dict:
        """
        Score a compression task using Vidaio's exact formulas.
        
        Args:
            reference_path: Path to reference video
            processed_path: Path to processed video
            vmaf_threshold: Required VMAF threshold
            
        Returns:
            Dictionary with all scores and metrics
        """
        print("🎬 Calculating VMAF score...")
        vmaf_score = self.calculate_vmaf_score(reference_path, processed_path)
        
        print("📊 Calculating compression rate...")
        ref_size = os.path.getsize(reference_path)
        proc_size = os.path.getsize(processed_path)
        compression_rate = proc_size / ref_size
        
        print("🔍 Calculating compression score...")
        s_f, compression_component, quality_component, reason = self.calculate_compression_score(
            vmaf_score, compression_rate, vmaf_threshold
        )
        
        return {
            "task_type": "compression",
            "vmaf_score": vmaf_score,
            "compression_rate": compression_rate,
            "compression_ratio": 1 / compression_rate,
            "s_f": s_f,
            "compression_component": compression_component,
            "quality_component": quality_component,
            "reason": reason,
            "vmaf_threshold": vmaf_threshold,
            "vmaf_threshold_passed": vmaf_score >= vmaf_threshold
        }

def load_performance_history(history_folder: str) -> List[Dict]:
    """
    Load performance history from JSON files in a folder.
    
    Expected file format: performance_round_XXX.json
    """
    history = []
    history_path = Path(history_folder)
    
    if not history_path.exists():
        print(f"History folder not found: {history_folder}")
        return []
    
    # Find all performance files
    performance_files = sorted(history_path.glob("performance_round_*.json"))
    
    for file_path in performance_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    history.extend(data)
                else:
                    history.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return history

def main():
    parser = argparse.ArgumentParser(description="Vidaio Local Validation Tool")
    parser.add_argument("--task", choices=["upscaling", "compression"], required=True,
                       help="Task type to score")
    parser.add_argument("--reference", required=True,
                       help="Path to reference video")
    parser.add_argument("--processed", required=True,
                       help="Path to processed video")
    parser.add_argument("--content-length", type=float, default=10.0,
                       help="Content length in seconds (for upscaling)")
    parser.add_argument("--threshold", type=float, default=85.0,
                       help="VMAF threshold (for compression)")
    parser.add_argument("--calculate-multipliers", action="store_true",
                       help="Calculate performance multipliers from history")
    parser.add_argument("--history-folder", default="./performance_history",
                       help="Folder containing performance history JSON files")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.reference):
        print(f"Error: Reference video not found: {args.reference}")
        return 1
    
    if not os.path.exists(args.processed):
        print(f"Error: Processed video not found: {args.processed}")
        return 1
    
    scorer = VidaioScorer()
    
    # Score the task
    if args.task == "upscaling":
        print("🎯 Scoring upscaling task...")
        results = scorer.score_upscaling_task(args.reference, args.processed, args.content_length)
    else:
        print("🎯 Scoring compression task...")
        results = scorer.score_compression_task(args.reference, args.processed, args.threshold)
    
    # Calculate multipliers if requested
    if args.calculate_multipliers:
        print("📊 Loading performance history...")
        history = load_performance_history(args.history_folder)
        multipliers = scorer.calculate_performance_multipliers(history)
        results["multipliers"] = multipliers
        results["history_count"] = len(history)
    
    # Print results
    print("\n" + "="*60)
    print("🎯 VIDAIO SCORING RESULTS")
    print("="*60)
    
    if args.task == "upscaling":
        print(f"📹 Task Type: Upscaling")
        print(f"🎬 VMAF Score: {results['vmaf_score']:.2f} (threshold: {scorer.VMAF_THRESHOLD*100:.0f})")
        print(f"🎨 PIE-APP Score: {results['pieapp_score']:.4f} (threshold: {scorer.PIEAPP_THRESHOLD})")
        print(f"📏 Content Length: {results['content_length']:.1f}s")
        print(f"🔍 Quality Score (S_Q): {results['s_q']:.4f}")
        print(f"📏 Length Score (S_L): {results['s_l']:.4f}")
        print(f"⚡ Final Score (S_F): {results['s_f']:.4f}")
        print(f"✅ VMAF Threshold Passed: {results['vmaf_threshold_passed']}")
        print(f"✅ PIE-APP Threshold Passed: {results['pieapp_threshold_passed']}")
    else:
        print(f"📹 Task Type: Compression")
        print(f"🎬 VMAF Score: {results['vmaf_score']:.2f} (threshold: {results['vmaf_threshold']:.0f})")
        print(f"📊 Compression Rate: {results['compression_rate']:.4f} ({results['compression_ratio']:.2f}x)")
        print(f"🔍 Final Score (S_f): {results['s_f']:.4f}")
        print(f"📈 Compression Component: {results['compression_component']:.4f}")
        print(f"🎯 Quality Component: {results['quality_component']:.4f}")
        print(f"📝 Reason: {results['reason']}")
        print(f"✅ VMAF Threshold Passed: {results['vmaf_threshold_passed']}")
    
    if "multipliers" in results:
        print(f"\n📊 PERFORMANCE MULTIPLIERS (from {results['history_count']} rounds)")
        print(f"🎁 Bonus Multiplier: {results['multipliers']['bonus_multiplier']:.4f} ({results['multipliers']['bonus_count']}/10 excellent)")
        print(f"⚠️  Performance Penalty: {results['multipliers']['penalty_f_multiplier']:.4f} ({results['multipliers']['penalty_f_count']}/10 poor)")
        print(f"🚫 Quality Penalty: {results['multipliers']['penalty_q_multiplier']:.4f} ({results['multipliers']['penalty_q_count']}/10 failed)")
        print(f"⚡ Total Multiplier: {results['multipliers']['total_multiplier']:.4f}")
    
    # Performance tier
    if args.task == "upscaling":
        s_f = results['s_f']
        if s_f > 0.7:
            tier = "Elite (16.3× multiplier)"
        elif s_f > 0.6:
            tier = "Good (2.0× multiplier)"
        elif s_f > 0.5:
            tier = "Average (1.5× multiplier)"
        else:
            tier = "Poor (1.0× multiplier)"
        print(f"\n🏆 Performance Tier: {tier}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

