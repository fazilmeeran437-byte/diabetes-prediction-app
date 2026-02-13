# food_module.py

# Database of common foods with estimated glucose impact (example values)
food_db = {
    "White Rice": 35,
    "Brown Rice": 20,
    "Chapati": 15,
    "Apple": 10,
    "Banana": 12,
    "Soft Drink": 40,
    "Ice Cream": 30,
    "Eggs": 5,
    "Chicken": 5,
    "Vegetables": 5
}

def predict_food_impact(food_name):
    """
    Returns estimated glucose impact of a food item.
    If food is not in database, returns 0.
    """
    return food_db.get(food_name, 0)