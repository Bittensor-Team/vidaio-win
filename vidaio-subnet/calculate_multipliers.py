#!/usr/bin/env python3
"""
Vidaio Performance Multiplier Calculator

This script calculates performance multipliers from a folder of performance history files.
It implements the exact Vidaio formulas for bonus/penalty calculations.

Usage:
    python calculate_multipliers.py --history-folder ./performance_history
    python calculate_multipliers.py --history-folder ./performance_history --output results.json
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict
from datetime import datetime

class VidaioMultiplierCalculator:
    """
    Calculates Vidaio performance multipliers using exact formulas.
    """
    
    def __init__(self):
        # Performance multiplier constants from vidaio_subnet_core/validating/managing/miner_manager.py
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

    def load_performance_history(self, history_folder: str) -> List[Dict]:
        """
        Load performance history from JSON files in a folder.
        
        Expected file format: performance_round_XXX.json or any JSON files
        """
        history = []
        history_path = Path(history_folder)
        
        if not history_path.exists():
            print(f"❌ History folder not found: {history_folder}")
            return []
        
        # Find all JSON files
        json_files = sorted(history_path.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in: {history_folder}")
            return []
        
        print(f"📁 Found {len(json_files)} JSON files in {history_folder}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        history.extend(data)
                        print(f"  📄 Loaded {len(data)} records from {file_path.name}")
                    else:
                        history.append(data)
                        print(f"  📄 Loaded 1 record from {file_path.name}")
            except Exception as e:
                print(f"  ❌ Error loading {file_path.name}: {e}")
        
        print(f"📊 Total records loaded: {len(history)}")
        return history

    def calculate_multipliers(self, performance_history: List[Dict]) -> Dict:
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
            Dictionary with multiplier values and statistics
        """
        if len(performance_history) == 0:
            return {
                "bonus_multiplier": 1.0,
                "penalty_f_multiplier": 1.0,
                "penalty_q_multiplier": 1.0,
                "total_multiplier": 1.0,
                "bonus_count": 0,
                "penalty_f_count": 0,
                "penalty_q_count": 0,
                "total_rounds": 0,
                "recent_rounds": 0,
                "message": "No performance history available"
            }
        
        # Use last 10 rounds (rolling window)
        recent_records = performance_history[-10:]
        total_rounds = len(performance_history)
        recent_rounds = len(recent_records)
        
        print(f"📈 Analyzing {recent_rounds} recent rounds (out of {total_rounds} total)")
        
        # Count bonuses and penalties
        bonus_count = 0
        penalty_f_count = 0
        penalty_q_count = 0
        
        upscaling_rounds = 0
        compression_rounds = 0
        
        for i, record in enumerate(recent_records):
            s_f = record.get("s_f", 0.0)
            s_q = record.get("s_q", 0.0)
            vmaf_score = record.get("vmaf_score", 0.0)
            vmaf_threshold = record.get("vmaf_threshold", 0.0)
            task_type = record.get("task_type", "upscaling")
            
            if task_type == "upscaling":
                upscaling_rounds += 1
                # Upscaling thresholds
                if s_f > self.UPSCALING_BONUS_THRESHOLD:
                    bonus_count += 1
                    print(f"  🎁 Round {i+1}: Bonus! S_f={s_f:.4f} > {self.UPSCALING_BONUS_THRESHOLD}")
                if s_f < self.PENALTY_F_THRESHOLD:
                    penalty_f_count += 1
                    print(f"  ⚠️  Round {i+1}: Performance penalty! S_f={s_f:.4f} < {self.PENALTY_F_THRESHOLD}")
                if s_q < self.PENALTY_Q_THRESHOLD:
                    penalty_q_count += 1
                    print(f"  🚫 Round {i+1}: Quality penalty! S_q={s_q:.4f} < {self.PENALTY_Q_THRESHOLD}")
            else:
                compression_rounds += 1
                # Compression thresholds
                if s_f > self.COMPRESSION_BONUS_THRESHOLD:
                    bonus_count += 1
                    print(f"  🎁 Round {i+1}: Bonus! S_f={s_f:.4f} > {self.COMPRESSION_BONUS_THRESHOLD}")
                if s_f < self.COMPRESSION_PENALTY_F_THRESHOLD:
                    penalty_f_count += 1
                    print(f"  ⚠️  Round {i+1}: Performance penalty! S_f={s_f:.4f} < {self.COMPRESSION_PENALTY_F_THRESHOLD}")
                if vmaf_score < vmaf_threshold + self.COMPRESSION_VMAF_MARGIN:
                    penalty_q_count += 1
                    print(f"  🚫 Round {i+1}: Quality penalty! VMAF={vmaf_score:.2f} < {vmaf_threshold + self.COMPRESSION_VMAF_MARGIN}")
        
        # Calculate multipliers using exact Vidaio formulas
        bonus_multiplier = 1.0 + (bonus_count / 10) * self.BONUS_MAX
        penalty_f_multiplier = 1.0 - (penalty_f_count / 10) * self.PENALTY_F_MAX
        penalty_q_multiplier = 1.0 - (penalty_q_count / 10) * self.PENALTY_Q_MAX
        
        # Total multiplier is multiplicative
        total_multiplier = bonus_multiplier * penalty_f_multiplier * penalty_q_multiplier
        
        # Calculate performance tier
        if total_multiplier >= 1.15:
            tier = "Elite"
            tier_multiplier = 16.3
        elif total_multiplier >= 1.05:
            tier = "Good"
            tier_multiplier = 2.0
        elif total_multiplier >= 0.95:
            tier = "Average"
            tier_multiplier = 1.5
        else:
            tier = "Poor"
            tier_multiplier = 1.0
        
        return {
            "bonus_multiplier": bonus_multiplier,
            "penalty_f_multiplier": penalty_f_multiplier,
            "penalty_q_multiplier": penalty_q_multiplier,
            "total_multiplier": total_multiplier,
            "bonus_count": bonus_count,
            "penalty_f_count": penalty_f_count,
            "penalty_q_count": penalty_q_count,
            "total_rounds": total_rounds,
            "recent_rounds": recent_rounds,
            "upscaling_rounds": upscaling_rounds,
            "compression_rounds": compression_rounds,
            "performance_tier": tier,
            "tier_multiplier": tier_multiplier,
            "message": f"Analyzed {recent_rounds} recent rounds"
        }

    def analyze_performance_trends(self, performance_history: List[Dict]) -> Dict:
        """
        Analyze performance trends over time.
        """
        if len(performance_history) < 2:
            return {"message": "Not enough data for trend analysis"}
        
        # Calculate rolling averages
        window_size = min(10, len(performance_history))
        recent_scores = [record.get("s_f", 0.0) for record in performance_history[-window_size:]]
        older_scores = [record.get("s_f", 0.0) for record in performance_history[-window_size*2:-window_size]] if len(performance_history) >= window_size*2 else []
        
        recent_avg = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
        older_avg = sum(older_scores) / len(older_scores) if older_scores else recent_avg
        
        trend = "stable"
        if recent_avg > older_avg * 1.1:
            trend = "improving"
        elif recent_avg < older_avg * 0.9:
            trend = "declining"
        
        return {
            "recent_average_s_f": recent_avg,
            "older_average_s_f": older_avg,
            "trend": trend,
            "improvement_percentage": ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0.0
        }

def main():
    parser = argparse.ArgumentParser(description="Vidaio Performance Multiplier Calculator")
    parser.add_argument("--history-folder", required=True,
                       help="Folder containing performance history JSON files")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--trends", action="store_true",
                       help="Include performance trend analysis")
    
    args = parser.parse_args()
    
    calculator = VidaioMultiplierCalculator()
    
    # Load performance history
    print("📊 Loading performance history...")
    history = calculator.load_performance_history(args.history_folder)
    
    if not history:
        print("❌ No performance history found. Exiting.")
        return 1
    
    # Calculate multipliers
    print("\n🔍 Calculating performance multipliers...")
    results = calculator.calculate_multipliers(history)
    
    # Add trend analysis if requested
    if args.trends:
        print("\n📈 Analyzing performance trends...")
        trends = calculator.analyze_performance_trends(history)
        results["trends"] = trends
    
    # Print results
    print("\n" + "="*70)
    print("🎯 VIDAIO PERFORMANCE MULTIPLIER ANALYSIS")
    print("="*70)
    
    print(f"📊 Data Summary:")
    print(f"  Total rounds: {results['total_rounds']}")
    print(f"  Recent rounds analyzed: {results['recent_rounds']}")
    print(f"  Upscaling rounds: {results['upscaling_rounds']}")
    print(f"  Compression rounds: {results['compression_rounds']}")
    
    print(f"\n🎁 Bonus Analysis:")
    print(f"  Excellent rounds: {results['bonus_count']}/10")
    print(f"  Bonus multiplier: {results['bonus_multiplier']:.4f}")
    print(f"  Max possible: {1.0 + self.BONUS_MAX:.4f}")
    
    print(f"\n⚠️  Penalty Analysis:")
    print(f"  Poor performance rounds: {results['penalty_f_count']}/10")
    print(f"  Performance penalty: {results['penalty_f_multiplier']:.4f}")
    print(f"  Quality failure rounds: {results['penalty_q_count']}/10")
    print(f"  Quality penalty: {results['penalty_q_multiplier']:.4f}")
    
    print(f"\n⚡ Final Multipliers:")
    print(f"  Total multiplier: {results['total_multiplier']:.4f}")
    print(f"  Performance tier: {results['performance_tier']}")
    print(f"  Tier multiplier: {results['tier_multiplier']}×")
    
    # Performance tier explanation
    if results['performance_tier'] == "Elite":
        print(f"\n🏆 ELITE PERFORMANCE!")
        print(f"   You're in the top tier with {results['tier_multiplier']}× multiplier")
        print(f"   This means you earn {results['tier_multiplier']}× more rewards than poor miners")
    elif results['performance_tier'] == "Good":
        print(f"\n✅ GOOD PERFORMANCE")
        print(f"   You're performing well with {results['tier_multiplier']}× multiplier")
        print(f"   Focus on consistency to reach Elite tier")
    elif results['performance_tier'] == "Average":
        print(f"\n⚠️  AVERAGE PERFORMANCE")
        print(f"   You're getting {results['tier_multiplier']}× multiplier")
        print(f"   Focus on improving quality and consistency")
    else:
        print(f"\n❌ POOR PERFORMANCE")
        print(f"   You're getting {results['tier_multiplier']}× multiplier")
        print(f"   This means you're earning much less than top miners")
        print(f"   Focus on improving video quality and processing consistency")
    
    # Trend analysis
    if args.trends and "trends" in results:
        trends = results["trends"]
        print(f"\n📈 Performance Trends:")
        print(f"  Recent average S_f: {trends['recent_average_s_f']:.4f}")
        print(f"  Previous average S_f: {trends['older_average_s_f']:.4f}")
        print(f"  Trend: {trends['trend']}")
        if trends['improvement_percentage'] != 0:
            print(f"  Change: {trends['improvement_percentage']:+.1f}%")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if results['bonus_count'] < 3:
        print(f"  🎁 Focus on achieving more excellent rounds (S_f > 0.32 for upscaling)")
    if results['penalty_f_count'] > 2:
        print(f"  ⚠️  Reduce poor performance rounds (S_f < 0.07)")
    if results['penalty_q_count'] > 2:
        print(f"  🚫 Improve quality consistency (S_q > 0.5 for upscaling)")
    if results['total_multiplier'] < 1.0:
        print(f"  🔧 Overall performance needs improvement")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

