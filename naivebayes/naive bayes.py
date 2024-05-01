import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Variables for chart plotting
categories = ["SalePrice", "SaleCondition", "SaleType", "YrSold", "MoSold"]
lst1 = []
lst2 = []

# User input values for relevant categories
#-------------------------------------------
cutoff = 0
val1 = input("Sale price estimate: ")
if val1:
    cutoff = int(val1)

sale_condition = ""
sale_type = ""
year = 0
month = 0
val2 = input("Sale Condition: ")
if val2:
    sale_condition = val2
val3 = input("Sale Type: ")
if val3:
    sale_type = val3
val4 = input("Year Sold: ")
if val4:
    year = int(val4)
val5 = input("Month Sold: ")
if val5:
    month = int(val5)
#-------------------------------------------

# Naive Bayes algorithm implementation
def naive_bayes():
    train_df = pd.read_csv(os.getcwd() + "/train.csv") # Uses training dataset

    # Counts # of items that are >= and < cutoff from SalePrice
    greater_equal = 0
    less = 0
    for price in train_df["SalePrice"]:
        if price >= cutoff:
            greater_equal = greater_equal + 1
        else:
            less = less + 1

    lst1.append(greater_equal)
    lst2.append(less)
    # Calculates >= and < prob ratios
    greater_equal_ratio = greater_equal / len(train_df["SalePrice"])
    less_ratio = less / len(train_df["SalePrice"])

    attributes = ["SaleCondition", "SaleType", "YrSold", "MoSold"]
    var_list = [sale_condition, sale_type, year, month]
    # Start of naive bayes formula for >= and <
    greater_equal_prob = greater_equal_ratio
    less_prob = less_ratio

    # Goes through every category and given user inputs
    for category, var in zip(attributes, var_list):
        n = 0
        m = 0
        # Counts # of items that are (1) same item as user input (2) >= and < cutoff
        for i, j in zip(train_df[category], train_df["SalePrice"]):
            if i == var and j >= cutoff:
                n = n + 1
            elif i == var and j < cutoff:
                m = m + 1

        lst1.append(n)
        lst2.append(m)
        # Calculates category prob conditioned to >= and <
        n_ratio = n / greater_equal
        m_ratio = m / less

        # Multiplies to their respective formulas
        greater_equal_prob = greater_equal_prob * n_ratio
        less_prob = less_prob * m_ratio

    print("\n>= prob: " + str(greater_equal_prob))
    print("< prob: " + str(less_prob))

    return greater_equal_prob, less_prob # Final prob calculations


# Chooses final prediction
num1, num2 = naive_bayes()
if num1 > num2:
    print("The sale price is predicted to be greater than or equal to " + str(cutoff))
elif num2 > num1:
    print("The sale price is predicted to be less than " + str(cutoff))
else:
    print("The algorithm can't determine a prediction")


# Creates bar chart of distributions for every category
X_axis = np.arange(len(categories)) 
plt.bar(X_axis - 0.2, lst1, 0.4, label = ">= " + str(cutoff)) 
plt.bar(X_axis + 0.2, lst2, 0.4, label = "< " + str(cutoff)) 
plt.xticks(X_axis, categories) 
plt.xlabel("Categories") 
plt.ylabel("# of Instances") 
plt.title("Distribution Based on Cutoff") 
plt.legend()
plt.savefig(os.path.join(os.getcwd(), "chart"))
plt.close()