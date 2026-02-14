#!/usr/bin/env python3
"""
Quick script to check order verification statistics.

Usage:
    python3 check_verification_stats.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket.performance.database import PerformanceDatabase
from datetime import datetime, timedelta

def main():
    db = PerformanceDatabase("data/performance.db")
    cursor = db.conn.cursor()
    
    print("=" * 60)
    print("ORDER VERIFICATION STATISTICS")
    print("=" * 60)
    
    # Overall verification status
    print("\n1. VERIFICATION STATUS")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            verification_status,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trades WHERE order_id IS NOT NULL), 1) as pct
        FROM trades
        WHERE order_id IS NOT NULL
        GROUP BY verification_status
        ORDER BY count DESC
    """)
    
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            status = row[0] or 'unverified'
            print(f"  {status:20s}: {row[1]:5d} trades ({row[2]:5.1f}%)")
    else:
        print("  No trades with order_id found yet")
    
    # Price discrepancies
    print("\n2. PRICE DISCREPANCIES")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            AVG(ABS(price_discrepancy_pct)) as avg_discrepancy,
            MAX(ABS(price_discrepancy_pct)) as max_discrepancy,
            COUNT(CASE WHEN ABS(price_discrepancy_pct) > 5.0 THEN 1 END) as large_discrepancies
        FROM trades
        WHERE verification_status = 'verified'
          AND price_discrepancy_pct IS NOT NULL
    """)
    
    row = cursor.fetchone()
    if row and row[0] > 0:
        print(f"  Total verified trades: {row[0]}")
        print(f"  Average discrepancy:   {row[1]:.2f}%")
        print(f"  Maximum discrepancy:   {row[2]:.2f}%")
        print(f"  Large discrepancies (>5%): {row[3]}")
    else:
        print("  No verified trades with price data yet")
    
    # Partial fills
    print("\n3. PARTIAL FILLS")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            COUNT(*) as total_partial,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM trades WHERE verification_status = 'verified'), 1) as pct
        FROM trades
        WHERE verification_status = 'verified'
          AND partial_fill = 1
    """)
    
    row = cursor.fetchone()
    if row and row[0] > 0:
        print(f"  Partial fills: {row[0]} ({row[1]}% of verified trades)")
    else:
        print("  No partial fills detected")
    
    # Failed verifications
    print("\n4. FAILED VERIFICATIONS")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            COUNT(*) as total_failed,
            skip_reason
        FROM trades
        WHERE verification_status = 'failed'
        GROUP BY skip_reason
    """)
    
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            reason = row[1] or 'Unknown'
            print(f"  {row[0]:3d} trades: {reason}")
    else:
        print("  No failed verifications")
    
    # Recent activity (last 24 hours)
    print("\n5. RECENT ACTIVITY (Last 24 hours)")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN verification_status = 'verified' THEN 1 END) as verified,
            COUNT(CASE WHEN verification_status = 'failed' THEN 1 END) as failed,
            COUNT(CASE WHEN verification_status IS NULL OR verification_status = 'unverified' THEN 1 END) as unverified
        FROM trades
        WHERE order_id IS NOT NULL
          AND timestamp > datetime('now', '-24 hours')
    """)
    
    row = cursor.fetchone()
    if row and row[0] > 0:
        print(f"  Total trades: {row[0]}")
        print(f"  Verified:     {row[1]} ({row[1]/row[0]*100:.1f}%)")
        print(f"  Failed:       {row[2]} ({row[2]/row[0]*100:.1f}%)")
        print(f"  Unverified:   {row[3]} ({row[3]/row[0]*100:.1f}%)")
    else:
        print("  No trades in last 24 hours")
    
    # P&L accuracy impact
    print("\n6. P&L ACCURACY IMPACT")
    print("-" * 60)
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            AVG(verified_fill_price - executed_price) as avg_price_diff,
            SUM(ABS(verified_fill_price - executed_price) * verified_fill_amount) as total_impact
        FROM trades
        WHERE verification_status = 'verified'
          AND verified_fill_price IS NOT NULL
          AND executed_price IS NOT NULL
    """)
    
    row = cursor.fetchone()
    if row and row[0] > 0:
        print(f"  Verified trades: {row[0]}")
        print(f"  Avg price difference: ${row[1]:.4f} per share")
        print(f"  Total $ impact prevented: ${row[2]:.2f}")
    else:
        print("  No verified trades with price comparison data yet")
    
    print("\n" + "=" * 60)
    print("Report generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

if __name__ == "__main__":
    main()
