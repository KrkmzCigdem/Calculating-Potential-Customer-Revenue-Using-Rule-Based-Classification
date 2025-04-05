#############################################
# Calculating Potential Customer Revenue Using Rule-Based Classification
#############################################

#############################################
# Business Problem
#############################################
# A gaming company wants to create new level-based customer definitions (personas)
# using some characteristics of its customers, and then form segments based on these
# personas to estimate the average revenue potential from future customers.

# For example: The company wants to determine the average revenue potential of a 25-year-old
# male user from Turkey using an iOS device.

#############################################
# Dataset Story
#############################################
# The persona.csv dataset contains the prices of products sold by an international gaming company
# and some demographic information about the users who bought these products.
# Each row represents a single sales transaction, meaning the table is not deduplicated.
# In other words, users with the same demographic attributes may appear more than once.

# Price: Customer's spending amount
# Source: Type of device the customer uses
# Sex: Customer's gender
# Country: Customer's country
# Age: Customer's age

################# Before Processing #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Processing #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

#############################################
# PROJECT TASKS
#############################################

#############################################
# TASK 1: Answer the following questions.
#############################################

import pandas as pd
pd.set_option("display.max_rows", None)

# Q1: Load persona.csv and show general information
df = pd.read_csv('persona.csv')
df.head()
df.shape
df.info()

# Q2: How many unique SOURCE values? What are their frequencies?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Q3: How many unique PRICE values?
df["PRICE"].nunique()

# Q4: How many sales for each PRICE?
df["PRICE"].value_counts()

# Q5: Number of sales per COUNTRY
df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="count")

# Q6: Total revenue per COUNTRY
df.groupby("COUNTRY")["PRICE"].sum()
df.pivot_table(values="PRICE", index="COUNTRY", aggfunc="sum")

# Q7: Sales count by SOURCE
df["SOURCE"].value_counts()

# Q8: Average PRICE by COUNTRY
df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})

# Q9: Average PRICE by SOURCE
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})

# Q10: Average PRICE by COUNTRY-SOURCE breakdown
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})

#############################################
# TASK 2: Average revenue by COUNTRY, SOURCE, SEX, AGE
#############################################
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#############################################
# TASK 3: Sort by PRICE
#############################################
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

#############################################
# TASK 4: Convert index names to variable names
#############################################
agg_df = agg_df.reset_index()
agg_df.head()

#############################################
# TASK 5: Convert AGE into a categorical variable and add it to agg_df
#############################################
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()

#############################################
# TASK 6: Create level-based new customer definitions and add to dataset
#############################################
agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" +
                                   row[2].upper() + "_" + row[5].upper()
                                   for row in agg_df.values]
agg_df = agg_df[["customers_level_based", "PRICE"]]

# Group by customer level and calculate average price
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"}).reset_index()
agg_df.head()

#############################################
# TASK 7: Segment the new customers
#############################################
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})

#############################################
# TASK 8: Classify new customers and predict revenue
#############################################
# A 33-year-old female from Turkey using ANDROID
new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])

# A 35-year-old female from France using IOS
new_user = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_user])
