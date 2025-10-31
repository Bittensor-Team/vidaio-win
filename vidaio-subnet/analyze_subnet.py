#!/usr/bin/env python3
"""
Vidaio Subnet Analysis Tool

This script analyzes the Vidaio subnet (netuid 85) using Bittensor API to:
1. Find top 10 miners by stake
2. Count active miners getting emissions
3. Analyze stake distribution
4. Check miner performance data

Usage:
    python analyze_subnet.py
    python analyze_subnet.py --output subnet_analysis.json
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, List, Optional
from datetime import datetime

# Add the project root to Python path
sys.path.append('/workspace/vidaio-subnet')

try:
    import bittensor as bt
    from bittensor import subtensor
except ImportError:
    print("❌ Bittensor not installed. Install with: pip install bittensor")
    sys.exit(1)

class VidaioSubnetAnalyzer:
    """
    Analyzes Vidaio subnet data using Bittensor API.
    """
    
    def __init__(self, network: str = "finney"):
        self.network = network
        self.netuid = 85  # Vidaio subnet
        self.subtensor = subtensor(network=network)
        
    def get_subnet_info(self) -> Dict:
        """
        Get basic subnet information.
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            return {
                "netuid": self.netuid,
                "network": self.network,
                "total_miners": len(metagraph.uids),
                "total_stake": float(metagraph.S.sum()),
                "average_stake": float(metagraph.S.mean()),
                "max_stake": float(metagraph.S.max()),
                "min_stake": float(metagraph.S.min()),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Failed to get subnet info: {e}"}
    
    def get_top_miners(self, top_n: int = 10) -> List[Dict]:
        """
        Get top N miners by stake (excluding validators).
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            # Create list of (uid, stake, hotkey, is_validator) tuples
            all_nodes = []
            for i, uid in enumerate(metagraph.uids):
                stake = float(metagraph.S[i])
                hotkey = metagraph.hotkeys[i]
                is_validator = bool(metagraph.validator_permit[i]) if hasattr(metagraph, 'validator_permit') else False
                all_nodes.append((uid, stake, hotkey, is_validator))
            
            # Filter out validators to get only miners
            miners = [(uid, stake, hotkey) for uid, stake, hotkey, is_validator in all_nodes if not is_validator]
            
            if not miners:
                return [{"error": "No miners found (all nodes are validators)"}]
            
            # Sort by stake (descending)
            miners.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N
            top_miners = []
            for i, (uid, stake, hotkey) in enumerate(miners[:top_n]):
                top_miners.append({
                    "rank": i + 1,
                    "uid": int(uid),
                    "stake": stake,
                    "hotkey": hotkey,
                    "stake_percentage": (stake / metagraph.S.sum()) * 100
                })
            
            return top_miners
            
        except Exception as e:
            return [{"error": f"Failed to get top miners: {e}"}]
    
    def get_active_miners(self) -> Dict:
        """
        Get information about active miners (those getting emissions, excluding validators).
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            # Separate miners and validators
            active_miners = []
            validators = []
            total_miner_stake = 0.0
            total_validator_stake = 0.0
            
            for i, uid in enumerate(metagraph.uids):
                stake = float(metagraph.S[i])
                hotkey = metagraph.hotkeys[i]
                is_validator = bool(metagraph.validator_permit[i]) if hasattr(metagraph, 'validator_permit') else False
                
                if stake > 0:
                    if is_validator:
                        validators.append({
                            "uid": int(uid),
                            "stake": stake,
                            "hotkey": hotkey
                        })
                        total_validator_stake += stake
                    else:
                        active_miners.append({
                            "uid": int(uid),
                            "stake": stake,
                            "hotkey": hotkey
                        })
                        total_miner_stake += stake
            
            return {
                "active_miner_count": len(active_miners),
                "total_miner_stake": total_miner_stake,
                "average_miner_stake": total_miner_stake / len(active_miners) if active_miners else 0,
                "active_miners": active_miners,
                "validator_count": len(validators),
                "total_validator_stake": total_validator_stake,
                "average_validator_stake": total_validator_stake / len(validators) if validators else 0,
                "validators": validators
            }
            
        except Exception as e:
            return {"error": f"Failed to get active miners: {e}"}
    
    def analyze_stake_distribution(self) -> Dict:
        """
        Analyze stake distribution across miners (excluding validators).
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            # Separate miner and validator stakes
            miner_stakes = []
            validator_stakes = []
            
            for i, stake in enumerate(metagraph.S):
                stake_val = float(stake)
                if stake_val > 0:
                    is_validator = bool(metagraph.validator_permit[i]) if hasattr(metagraph, 'validator_permit') else False
                    if is_validator:
                        validator_stakes.append(stake_val)
                    else:
                        miner_stakes.append(stake_val)
            
            if not miner_stakes:
                return {"error": "No miner stakes found"}
            
            miner_stakes.sort(reverse=True)
            
            # Calculate percentiles
            def percentile(data, p):
                k = (len(data) - 1) * p / 100
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return data[f] * (1 - c) + data[f + 1] * c
                else:
                    return data[f]
            
            return {
                "miner_stakes": {
                    "total_miners_with_stake": len(miner_stakes),
                    "total_stake": sum(miner_stakes),
                    "average_stake": sum(miner_stakes) / len(miner_stakes),
                    "median_stake": percentile(miner_stakes, 50),
                    "p90_stake": percentile(miner_stakes, 90),
                    "p95_stake": percentile(miner_stakes, 95),
                    "p99_stake": percentile(miner_stakes, 99),
                    "max_stake": max(miner_stakes),
                    "min_stake": min(miner_stakes),
                    "stake_std": (sum((x - sum(miner_stakes)/len(miner_stakes))**2 for x in miner_stakes) / len(miner_stakes))**0.5
                },
                "validator_stakes": {
                    "total_validators_with_stake": len(validator_stakes),
                    "total_stake": sum(validator_stakes),
                    "average_stake": sum(validator_stakes) / len(validator_stakes) if validator_stakes else 0,
                    "max_stake": max(validator_stakes) if validator_stakes else 0,
                    "min_stake": min(validator_stakes) if validator_stakes else 0
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze stake distribution: {e}"}
    
    def get_miner_performance_data(self, uid: int) -> Dict:
        """
        Get performance data for a specific miner.
        Note: This is limited by what's available through Bittensor API.
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            if uid not in metagraph.uids:
                return {"error": f"UID {uid} not found in subnet"}
            
            uid_index = list(metagraph.uids).index(uid)
            
            return {
                "uid": int(uid),
                "hotkey": metagraph.hotkeys[uid_index],
                "stake": float(metagraph.S[uid_index]),
                "rank": int(metagraph.R[uid_index]) if hasattr(metagraph, 'R') else None,
                "emission": float(metagraph.E[uid_index]) if hasattr(metagraph, 'E') else None,
                "incentive": float(metagraph.I[uid_index]) if hasattr(metagraph, 'I') else None,
                "consensus": float(metagraph.C[uid_index]) if hasattr(metagraph, 'C') else None,
                "trust": float(metagraph.T[uid_index]) if hasattr(metagraph, 'T') else None,
                "validator_permit": bool(metagraph.validator_permit[uid_index]) if hasattr(metagraph, 'validator_permit') else None
            }
            
        except Exception as e:
            return {"error": f"Failed to get miner data for UID {uid}: {e}"}
    
    def analyze_subnet_health(self) -> Dict:
        """
        Analyze overall subnet health and activity.
        """
        try:
            metagraph = self.subtensor.metagraph(netuid=self.netuid)
            
            # Basic metrics
            total_miners = len(metagraph.uids)
            active_miners = sum(1 for s in metagraph.S if s > 0)
            total_stake = float(metagraph.S.sum())
            
            # Calculate Gini coefficient for stake distribution
            stakes = [float(s) for s in metagraph.S if s > 0]
            if len(stakes) > 1:
                stakes.sort()
                n = len(stakes)
                cumsum = [sum(stakes[:i+1]) for i in range(n)]
                gini = (n + 1 - 2 * sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            else:
                gini = 0
            
            # Health indicators
            health_score = 0
            health_issues = []
            
            if active_miners < 10:
                health_score -= 20
                health_issues.append("Very few active miners")
            elif active_miners < 50:
                health_score -= 10
                health_issues.append("Low number of active miners")
            
            if total_stake < 1000:
                health_score -= 20
                health_issues.append("Very low total stake")
            elif total_stake < 10000:
                health_score -= 10
                health_issues.append("Low total stake")
            
            if gini > 0.8:
                health_score -= 15
                health_issues.append("High stake concentration (Gini > 0.8)")
            elif gini > 0.6:
                health_score -= 5
                health_issues.append("Moderate stake concentration (Gini > 0.6)")
            
            if health_score >= 80:
                health_status = "Excellent"
            elif health_score >= 60:
                health_status = "Good"
            elif health_score >= 40:
                health_status = "Fair"
            else:
                health_status = "Poor"
            
            return {
                "health_score": health_score,
                "health_status": health_status,
                "health_issues": health_issues,
                "total_miners": total_miners,
                "active_miners": active_miners,
                "total_stake": total_stake,
                "gini_coefficient": gini,
                "stake_concentration": "High" if gini > 0.6 else "Moderate" if gini > 0.4 else "Low"
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze subnet health: {e}"}

def main():
    parser = argparse.ArgumentParser(description="Vidaio Subnet Analysis Tool")
    parser.add_argument("--network", default="finney",
                       help="Bittensor network (default: finney)")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top miners to show (default: 10)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--miner-uid", type=int, help="Analyze specific miner UID")
    
    args = parser.parse_args()
    
    print("🔍 Initializing Vidaio Subnet Analysis...")
    print(f"Network: {args.network}")
    print(f"Subnet UID: 85 (Vidaio)")
    
    analyzer = VidaioSubnetAnalyzer(network=args.network)
    
    # Get subnet info
    print("\n📊 Getting subnet information...")
    subnet_info = analyzer.get_subnet_info()
    
    if "error" in subnet_info:
        print(f"❌ Error: {subnet_info['error']}")
        return 1
    
    # Get top miners
    print(f"\n🏆 Getting top {args.top_n} miners...")
    top_miners = analyzer.get_top_miners(args.top_n)
    
    # Get active miners
    print("\n⚡ Getting active miners...")
    active_miners = analyzer.get_active_miners()
    
    # Analyze stake distribution
    print("\n📈 Analyzing stake distribution...")
    stake_dist = analyzer.analyze_stake_distribution()
    
    # Analyze subnet health
    print("\n🏥 Analyzing subnet health...")
    health = analyzer.analyze_subnet_health()
    
    # Get specific miner data if requested
    miner_data = None
    if args.miner_uid:
        print(f"\n👤 Getting data for miner UID {args.miner_uid}...")
        miner_data = analyzer.get_miner_performance_data(args.miner_uid)
    
    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "subnet_info": subnet_info,
        "top_miners": top_miners,
        "active_miners": active_miners,
        "stake_distribution": stake_dist,
        "subnet_health": health,
        "miner_data": miner_data
    }
    
    # Print results
    print("\n" + "="*70)
    print("🎯 VIDAIO SUBNET ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n📊 Subnet Overview:")
    print(f"  Total miners: {subnet_info['total_miners']}")
    print(f"  Total stake: {subnet_info['total_stake']:,.2f} TAO")
    print(f"  Average stake: {subnet_info['average_stake']:,.2f} TAO")
    print(f"  Max stake: {subnet_info['max_stake']:,.2f} TAO")
    print(f"  Min stake: {subnet_info['min_stake']:,.2f} TAO")
    
    print(f"\n🏆 Top {args.top_n} Miners by Stake:")
    for miner in top_miners:
        if "error" not in miner:
            print(f"  #{miner['rank']:2d}: UID {miner['uid']:3d} - {miner['stake']:8,.2f} TAO ({miner['stake_percentage']:5.2f}%)")
        else:
            print(f"  Error: {miner['error']}")
    
    if "error" not in active_miners:
        print(f"\n⚡ Active Miners:")
        print(f"  Active miner count: {active_miners['active_miner_count']}")
        print(f"  Total miner stake: {active_miners['total_miner_stake']:,.2f} TAO")
        print(f"  Average miner stake: {active_miners['average_miner_stake']:,.2f} TAO")
        print(f"\n🔍 Validators:")
        print(f"  Validator count: {active_miners['validator_count']}")
        print(f"  Total validator stake: {active_miners['total_validator_stake']:,.2f} TAO")
        print(f"  Average validator stake: {active_miners['average_validator_stake']:,.2f} TAO")
    
    if "error" not in stake_dist:
        print(f"\n📈 Stake Distribution:")
        print(f"  Miners with stake: {stake_dist['miner_stakes']['total_miners_with_stake']}")
        print(f"  Median miner stake: {stake_dist['miner_stakes']['median_stake']:,.2f} TAO")
        print(f"  90th percentile: {stake_dist['miner_stakes']['p90_stake']:,.2f} TAO")
        print(f"  95th percentile: {stake_dist['miner_stakes']['p95_stake']:,.2f} TAO")
        print(f"  99th percentile: {stake_dist['miner_stakes']['p99_stake']:,.2f} TAO")
        print(f"  Standard deviation: {stake_dist['miner_stakes']['stake_std']:,.2f} TAO")
        print(f"\n🔍 Validator Stake Distribution:")
        print(f"  Validators with stake: {stake_dist['validator_stakes']['total_validators_with_stake']}")
        print(f"  Total validator stake: {stake_dist['validator_stakes']['total_stake']:,.2f} TAO")
        print(f"  Average validator stake: {stake_dist['validator_stakes']['average_stake']:,.2f} TAO")
    
    if "error" not in health:
        print(f"\n🏥 Subnet Health:")
        print(f"  Health score: {health['health_score']}/100")
        print(f"  Status: {health['health_status']}")
        print(f"  Gini coefficient: {health['gini_coefficient']:.3f}")
        print(f"  Stake concentration: {health['stake_concentration']}")
        if health['health_issues']:
            print(f"  Issues: {', '.join(health['health_issues'])}")
    
    if miner_data and "error" not in miner_data:
        print(f"\n👤 Miner UID {args.miner_uid} Data:")
        print(f"  Hotkey: {miner_data['hotkey']}")
        print(f"  Stake: {miner_data['stake']:,.2f} TAO")
        if miner_data['rank']:
            print(f"  Rank: {miner_data['rank']}")
        if miner_data['emission']:
            print(f"  Emission: {miner_data['emission']:.6f}")
        if miner_data['incentive']:
            print(f"  Incentive: {miner_data['incentive']:.6f}")
        if miner_data['consensus']:
            print(f"  Consensus: {miner_data['consensus']:.6f}")
        if miner_data['trust']:
            print(f"  Trust: {miner_data['trust']:.6f}")
        if miner_data['validator_permit'] is not None:
            print(f"  Validator permit: {miner_data['validator_permit']}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
