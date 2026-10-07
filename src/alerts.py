"""
Alert system and retrain triggers
"""

from config import RETRAIN_TRIGGER_CONSECUTIVE_RED


def determine_alert_level(drift_metrics):
    """
    Determine overall alert level from drift metrics

    Logic:
    - 🔴 RED: If ANY metric shows RED
    - 🟡 AMBER: If ANY metric shows AMBER (and none RED)
    - 🟢 GREEN: All metrics GREEN

    Args:
        drift_metrics: Dictionary with alert levels from PSI, KS, PR-AUC

    Returns:
        Overall alert level ('GREEN', 'AMBER', or 'RED')
    """
    alerts = []

    if 'psi_alert' in drift_metrics:
        alerts.append(drift_metrics['psi_alert'])
    if 'ks_alert' in drift_metrics:
        alerts.append(drift_metrics['ks_alert'])
    if 'performance_alert' in drift_metrics:
        alerts.append(drift_metrics['performance_alert'])

    # RED if any metric is RED
    if 'RED' in alerts:
        return 'RED'
    # AMBER if any metric is AMBER
    elif 'AMBER' in alerts:
        return 'AMBER'
    # GREEN otherwise
    else:
        return 'GREEN'


def check_retrain_trigger(window_alerts, consecutive_red_required=RETRAIN_TRIGGER_CONSECUTIVE_RED):
    """
    Check if retrain should be triggered

    Retrain Trigger:
    - 2 consecutive RED windows detected

    Why 2 consecutive?
    - Avoids false alarms from one noisy batch
    - Confirms persistent drift, not random fluctuation

    Args:
        window_alerts: List of alert levels for each window ['GREEN', 'RED', 'RED']
        consecutive_red_required: Number of consecutive REDs needed (default 2)

    Returns:
        Boolean - True if retrain should be triggered
    """
    if len(window_alerts) < consecutive_red_required:
        return False

    # Check for consecutive REDs
    for i in range(len(window_alerts) - consecutive_red_required + 1):
        window_slice = window_alerts[i:i + consecutive_red_required]
        if all(alert == 'RED' for alert in window_slice):
            return True

    return False


def generate_alert_report(window_results):
    """
    Generate human-readable alert report

    Args:
        window_results: List of window analysis dictionaries

    Returns:
        Formatted string report
    """
    report = []
    report.append("="*70)
    report.append("DRIFT MONITORING ALERT REPORT")
    report.append("="*70)
    report.append("")

    for i, window in enumerate(window_results, 1):
        alert = window.get('alert_level', 'UNKNOWN')
        emoji = "🔴" if alert == "RED" else "🟡" if alert == "AMBER" else "🟢"

        report.append(f"{emoji} Window {i}: {alert}")
        report.append(f"  PR-AUC: {window.get('pr_auc', 0):.4f}")
        report.append(f"  Drop: {window.get('pr_auc_drop_pct', 0):.2f}%")

        if 'max_psi' in window:
            report.append(f"  Max PSI: {window['max_psi']:.4f}")
        if 'drift_features_count' in window:
            report.append(f"  Features with drift: {window['drift_features_count']}")

        report.append("")

    # Retrain recommendation
    window_alerts = [w.get('alert_level') for w in window_results]
    should_retrain = check_retrain_trigger(window_alerts)

    report.append("-"*70)
    if should_retrain:
        report.append("⚠️  RETRAIN TRIGGERED: 2 consecutive RED windows detected")
        report.append("    Action Required: Retrain model on recent data")
    else:
        report.append("✓  No retrain needed at this time")

    report.append("="*70)

    return "\n".join(report)


def get_alert_emoji(alert_level):
    """
    Get emoji for alert level

    Args:
        alert_level: 'GREEN', 'AMBER', or 'RED'

    Returns:
        Emoji string
    """
    if alert_level == 'RED':
        return '🔴'
    elif alert_level == 'AMBER':
        return '🟡'
    else:
        return '🟢'
