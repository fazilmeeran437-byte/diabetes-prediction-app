# health_suggestions.py

def suggest(glucose, steps, sleep, stress):
    """
    Returns a list of AI health suggestions
    based on current glucose and lifestyle factors.
    """
    tips = []

    # Glucose-based suggestions
    if glucose > 140:
        tips.append("Reduce sugar intake")
    elif glucose < 80:
        tips.append("Consider small healthy snack")

    # Steps
    if steps < 4000:
        tips.append("Walk at least 30 minutes today")

    # Sleep
    if sleep < 6:
        tips.append("Try to sleep at least 7 hours")

    # Stress
    if stress > 7:
        tips.append("Do relaxation exercises or meditation")

    # Default
    if not tips:
        tips.append("Great! Maintain your healthy lifestyle")

    return tips