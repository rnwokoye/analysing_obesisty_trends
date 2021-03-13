# analysing_obesisty_trends
# Dependencies and Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import gmaps
import os
import json 
import scipy.stats as st
from sklearn.linear_model import LinearRegression
import urllib
from scipy.stats import linregress
import time
from api_keys import g_key

# File to Load
file_to_load = "Resources/BRFSS__Table_of_Overweight_and_Obesity__BMI_.csv"

# Read Obesity CSV and store into Pandas data frame
obesity_df_full = pd.read_csv(file_to_load)
obesity_df_full.head(10)

# Extract Columns that will be used in analysis
reduced_obesity_df = obesity_df_full.loc[:, ["Year", "Locationabbr", "Response", "Break_Out", "Break_Out_Category", 
                                             "Sample_Size", "BreakoutID", "GeoLocation"]]
reduced_obesity_df.head()



# Rename columns for clarity
reduced_obesity_df = reduced_obesity_df.rename(columns={"Locationabbr": "State", 
                                                        "Response": "BMI Range", "Break_Out": "Class", 
                                                        "Break_Out_Category": "Class Category", 
                                                        "Sample_Size": "Number of Respondents",
                                                        "BreakoutID": "Class ID"})

reduced_obesity_df.head()

# Reorder columns for clairty
reduced_obesity_df = reduced_obesity_df[['Year', 'State', 'Number of Respondents', 
                                         'BMI Range', 'Class', 'Class Category', 'Class ID', 'GeoLocation']]
reduced_obesity_df.head()

# Split GeoLocation into lat and lng for loop
new_column = reduced_obesity_df["GeoLocation"].str.split(",", n = 1, expand = True) 
reduced_obesity_df["Lat"]= new_column[0] 
reduced_obesity_df["Lng"]= new_column[1] 
reduced_obesity_df.head()

# Remove unwanted characters from the Lat and Lng Columns
reduced_obesity_df['Lat'] = reduced_obesity_df['Lat'].str.replace('(', '')
reduced_obesity_df['Lng'] = reduced_obesity_df['Lng'].str.replace(')', '')

# Remove Geolocation
reduced_obesity_df = reduced_obesity_df.drop(columns=['GeoLocation'])

# Remove commas from Number of Respondents
reduced_obesity_df['Number of Respondents'] = reduced_obesity_df['Number of Respondents'].str.replace(',', '')

reduced_obesity_df['BMI Range'].unique()

# Convert Number of Respondents Columns to integer
reduced_obesity_df['Number of Respondents'] = pd.to_numeric(reduced_obesity_df['Number of Respondents'])
reduced_obesity_df.info()

# Keep only most recent year (2019)
recent_df = reduced_obesity_df.loc[reduced_obesity_df['Year'] == 2019]

# Drop year column
recent_df = recent_df.drop(columns=['Year'])

# Keep only number of Respondents and BMI Range for state v. state comparison
recent_df_reduced = recent_df.loc[:, ["State", "Number of Respondents", "BMI Range"]]

# Display max rows
pd.set_option('display.max_rows', None)

# Display the total number of states
total_states = len(recent_df_reduced["State"].unique())
total_states

# Drop the US Rows
drop_us = recent_df_reduced.drop(recent_df_reduced.index[4680:4684])

# Drop the UW Rows
drop_uw = drop_us.drop(recent_df_reduced.index[4788:4792])

# Reset index and Rename DF
clean_state_df = drop_uw.reset_index(drop=True)

# Check new length of states (should be 52, 50 states plus PR and Guam)
new_total_states = len(clean_state_df["State"].unique())
new_total_states

# Display the total number of respondents
respondent_total = clean_state_df["Number of Respondents"].sum()
respondent_total

# Get sum of total respondents for each state
state_sum = clean_state_df.groupby('State')['Number of Respondents'].sum()

# Get sum of total overweight respondents for each state
overweight_sum = clean_state_df.loc[clean_state_df["BMI Range"] == 'Overweight (BMI 25.0-29.9)'].groupby("State")["Number of Respondents"].sum()

# Find overweight rates for each state
overweight_rates = (overweight_sum / state_sum) * 100

# Get sum of total underweight respondents for each state
underweight_sum = clean_state_df.loc[clean_state_df["BMI Range"] == 'Underweight (BMI 12.0-18.4)'].groupby("State")["Number of Respondents"].sum()

# Find underweight rates for each state
underweight_rates = (underweight_sum / state_sum) * 100

# Get sum of total obese respondents for each state
obesity_sum = clean_state_df.loc[clean_state_df["BMI Range"] == 'Obese (BMI 30.0 - 99.8)'].groupby("State")["Number of Respondents"].sum()

# Find obesity rates for each state
obesity_rates = (obesity_sum / state_sum) * 100

# Get sum of total normalweight respondents for each state
normal_sum = clean_state_df.loc[clean_state_df["BMI Range"] == 'Normal Weight (BMI 18.5-24.9)'].groupby("State")["Number of Respondents"].sum()

# Find Obesity Rates for each state
normal_rates = (normal_sum / state_sum) * 100

# Create New Data Frame to hold the values

state_summary_df = pd.DataFrame({
                                "% Overweight": overweight_rates,
                                "% Underweight": underweight_rates,
                                "% Obese": obesity_rates,
                                "% Normal": normal_rates
                                                        })
state_summary_df.style.format({
                                "% Overweight":"{:,.2f}",
                                "% Underweight": "{:,.2f}",
                                "% Obese": "{:,.2f}",
                                "% Normal": "{:,.2f}"
                                   })

# Create Bar Graph Representing Overweight Rates
x_axis = recent_df_reduced['State'].unique()
y_axis = overweight_rates
state_obesity_bar = y_axis.plot(kind="bar", figsize=(15,5), title="2019 Overweight Rates by State")
state_obesity_bar.set_xlabel("State")
state_obesity_bar.set_ylabel("Percentage of Overweight Population")
plt.savefig("Output_Data/Fig1.png")

# Create Bar Graph Representing Underweight Rates
x_axis = recent_df_reduced['State'].unique()
y_axis = underweight_rates
state_obesity_bar = y_axis.plot(kind="bar", figsize=(15,5), title="2019 Underweight Rates by State")
state_obesity_bar.set_xlabel("State")
state_obesity_bar.set_ylabel("Percentage of Underweight Population")
plt.savefig("Output_Data/Fig2.png")

# Create Bar Graph Representing Overweight Rates
x_axis = reduced_obesity_df['State'].unique()
y_axis = obesity_rates
state_obesity_bar = y_axis.plot(kind="bar", figsize=(15,5), title="2019 Obese Rates by State")
state_obesity_bar.set_xlabel("State")
state_obesity_bar.set_ylabel("Percentage of Obese Population")
plt.savefig("Output_Data/Fig3.png")

# Create Bar Graph Representing Normal Rates
x_axis = recent_df_reduced['State'].unique()
y_axis = normal_rates
state_obesity_bar = y_axis.plot(kind="bar", figsize=(15,5), title="2019 Normal Weight Rates by State")
state_obesity_bar.set_xlabel("State")
state_obesity_bar.set_ylabel("Percentage of Normal Weight Population")
plt.savefig("Output_Data/Fig4.png")

# Check Data
reduced_obesity_df.head()

# Keep only Maryland data
md_df = reduced_obesity_df.loc[reduced_obesity_df['State'] == 'MD']

# Drop state, lat, and lng columns
md_df_reduced_1 = md_df.drop(columns=['State', 'Lat', 'Lng'])

# Bug Fix 
md_df_reduced_1['BMI Range'] = md_df_reduced_1['BMI Range'].str.replace('bmi', 'BMI')

# Reset index and Rename DF
md_clean_df_1 = md_df_reduced_1.reset_index(drop=True)
md_clean_df_1.head()

md_clean_df_1['Class'].unique()

# Get sum of total respondents for each year
md_year_sum = md_clean_df_1.groupby('Year')['Number of Respondents'].sum()

# Get sum of obese respondents for each year
md_obesity_sum = md_clean_df_1.loc[md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)'].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for each year
md_obesity_rates = (md_obesity_sum / md_year_sum) * 100

# Sort Values to Start with 2011
md_clean_df_1_ascending = md_clean_df_1.sort_values("Year", ascending=True)

# Plot Maryland obesity rates since 2011
x_values_md_line = md_clean_df_1_ascending['Year'].unique()
y_values_md_line = md_obesity_rates 
md_plot = plt.plot(x_values_md_line, y_values_md_line, color="blue") 
plt.title("Maryland")
plt.xlabel("Year")
plt.ylabel("Obesity Rates: % of Population")
plt.savefig("Output_Data/Fig5.png")

# Get sum of md black respondents for each year
md_black_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == 'Black, non-Hispanic'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md black obese respondents each year
md_black_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == 'Black, non-Hispanic')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for each year
md_black_obesity_rates = (md_black_obese / md_black_sum) * 100

# Plot Maryland black obesity rates since 2011
x_values_md_line = md_clean_df_1_ascending['Year'].unique()
y_values_black_line = md_black_obesity_rates
md_black_plot = plt.plot(x_values_md_line, y_values_black_line, color="blue") 
plt.title("Maryland: Black Non Hispanic")
plt.xlabel("Year")
plt.ylabel("Obesity Rates: % of Population")
plt.savefig("Output_Data/Fig6.png")

# Get sum of md low income respondents for each year
md_low_income_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == 'Less than $15,000'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md black obese respondents each year
md_low_income_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == 'Less than $15,000')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for each year
md_low_income_obesity_rates = (md_low_income_obese / md_low_income_sum) * 100

# Plot Maryland low income obesity rates since 2011
x_values_md_line = md_clean_df_1_ascending['Year'].unique()
y_values_low_income_line = md_low_income_obesity_rates
md__low_income_plot = plt.plot(x_values_md_line, y_values_low_income_line, color="blue") 
plt.title("Maryland: Low Income")
plt.xlabel("Year")
plt.ylabel("Obesity Rates: % of Population")
plt.savefig("Output_Data/Fig7.png")

# Get sum of md low education respondents for each year
md_low_ed_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == 'Less than H.S.'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md black obese respondents each year
md_low_ed_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == 'Less than H.S.')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for each year
md_low_ed_obesity_rates = (md_low_ed_obese / md_low_ed_sum) * 100

# Plot Maryland low education obesity rates since 2011
x_values_md_line = md_clean_df_1_ascending['Year'].unique()
y_values_low_education_line = md_low_ed_obesity_rates
md_low_ed_plot = plt.plot(x_values_md_line, y_values_low_education_line, color="blue") 
plt.title("Maryland: Less Than HS Education")
plt.xlabel("Year")
plt.ylabel("Obesity Rates: % of Population")
plt.savefig("Output_Data/Fig8.png")

# Get sum of md young adult respondents for each year
md_young_adult_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == '18-24'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md young adult obese respondents each year
md_young_adult_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == '18-24')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for each year
md_young_adult_obesity_rates = (md_young_adult_obese / md_young_adult_sum ) * 100

# Plot Maryland young adult obesity rates since 2011
x_values_md_line = md_clean_df_1_ascending['Year'].unique()
y_values_young_adult_line = md_young_adult_obesity_rates
md_young_adult_plot = plt.plot(x_values_md_line, y_values_young_adult_line, color="blue") 
plt.title("Maryland: Young Adult -- 18-24")
plt.xlabel("Year")
plt.ylabel("Obesity Rates: % of Population")
plt.savefig("Output_Data/Fig9.png")

# Get sum of total male respondents for each year
md_male_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == 'Male'].groupby("Year")["Number of Respondents"].sum()

# Find sum of male obese respondents each year
md_male_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == 'Male')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for males for each year
md_male_obesity_rates = (md_male_obese / md_male_sum) * 100

# Get sum of female respondents for each year
md_female_sum = md_clean_df_1.loc[md_clean_df_1["Class"] == 'Female'].groupby("Year")["Number of Respondents"].sum()

# Find sum of female obese respondents each year
md_female_obese = md_clean_df_1.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_df_1["Class"] == 'Female')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for females for each year
md_female_obesity_rates = (md_female_obese / md_female_sum) * 100

# Plot Male v. Female Obesity Rates since 2011
# line 1 points
x1 = x_values_md_line 
y1 = md_male_obesity_rates
plt.plot(x1, y1, label = "Male")
# line 2 points
x2 = x_values_md_line 
y2 = md_female_obesity_rates
# plotting the line 2 points 
plt.plot(x2, y2, label = "Female")
plt.xlabel('Year')
# Set the y axis label of the current axis.
plt.ylabel('Obesity Rates: % of Population')
# Set a title of the current axes.
plt.title('Maryland Male v. Female Obesity')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
plt.savefig("Output_Data/Fig10.png")

# Statistical Analysis
md_clean_df_1

# Keep only most recent year (2019)
md_clean_recent = md_clean_df_1.loc[md_clean_df_1['Year'] == 2019]

# Find obesity rates by group
md_sum_recent = md_clean_recent['Number of Respondents'].sum()
md_obesity_sum_recent = md_clean_recent.loc[md_clean_recent["BMI Range"] == 'Obese (BMI 30.0 - 99.8)'].groupby("Year")["Number of Respondents"].sum()

# md total obesity rate 
md_obesity_recent = (md_obesity_sum_recent / md_sum_recent) * 100
md_obesity_recent

# find income disparity

# Get sum of md low income respondents for 2019
md_low_income_sum_recent = md_clean_recent.loc[md_clean_recent["Class"] == 'Less than $15,000'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md low income respondents for 2019
md_low_income_obese_recent = md_clean_recent.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_recent["Class"] == 'Less than $15,000')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for low income 2019
md_low_income_obesity_rates_recent = (md_low_income_obese_recent / md_low_income_sum_recent) * 100

# Get sum of md low income respondents for 2019
md_mid_income_sum_recent = md_clean_recent.loc[md_clean_recent["Class"] == '$15,000-$24,999'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md low income respondents for 2019
md_mid_income_obese_recent = md_clean_recent.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_recent["Class"] == '$15,000-$24,999')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates for low income 2019
md_mid_income_obesity_rates_recent = (md_mid_income_obese_recent / md_mid_income_sum_recent) * 100

# Get sum of md high income respondents for 2019
md_high_income_sum_recent = md_clean_recent.loc[md_clean_recent["Class"] == '$25,000-$34,999'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md high income respondents for 2019
md_high_income_obese_recent = md_clean_recent.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_recent["Class"] == '$25,000-$34,999')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates high income 2019
md_high_income_obesity_rates_recent = (md_high_income_obese_recent / md_high_income_sum_recent) * 100

# Get sum of md higher income respondents for 2019
md_higher_income_sum_recent = md_clean_recent.loc[md_clean_recent["Class"] == '$35,000-$49,999'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md higher income respondents for 2019
md_higher_income_obese_recent = md_clean_recent.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_recent["Class"] == '$35,000-$49,999')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates higher income 2019
md_higher_income_obesity_rates_recent = (md_higher_income_obese_recent / md_higher_income_sum_recent) * 100

# Get sum of md max income respondents for 2019
md_max_income_sum_recent = md_clean_recent.loc[md_clean_recent["Class"] == '$50,000+'].groupby("Year")["Number of Respondents"].sum()

# Find sum of md max income respondents for 2019
md_max_income_obese_recent = md_clean_recent.loc[(md_clean_df_1["BMI Range"] == 'Obese (BMI 30.0 - 99.8)') & (md_clean_recent["Class"] == '$50,000+')].groupby("Year")["Number of Respondents"].sum()

# Get obesity rates max income 2019
md_max_income_obesity_rates_recent = (md_max_income_obese_recent / md_max_income_sum_recent) * 100

income_df = pd.DataFrame({"Poor": md_low_income_obesity_rates_recent,
                          "Low": md_mid_income_obesity_rates_recent,
                          "Mid": md_high_income_obesity_rates_recent,
                          "Upper-Mid": md_higher_income_obesity_rates_recent,
                          "High": md_max_income_obesity_rates_recent
                                                })
income_df 

# Create Bar Graph Representing Obesity Income Rates
y_axis = income_df[income_df.columns]
state_obesity_bar = y_axis.plot(kind="bar", figsize=(15,5), title="Obesity Rates by Income")
state_obesity_bar.set_ylabel("Obesity Rate")
plt.savefig("Output_Data/Fig11.png")

# Bug Fix: Drop US Rows
recent_2_df = recent_df.loc[reduced_obesity_df['State'] != 'US']

# Create new DF to hold values State, Lat, and Lng Values
location_df_2 = pd.DataFrame({"State": recent_2_df['State'].unique().tolist(),
                            "Lat": recent_2_df['Lat'].unique().tolist(),
                            "Lng": recent_2_df['Lng'].unique().tolist()
                                                })

# Drop Nan Values
location_df_3 = location_df_2.dropna()

location_df_4 = location_df_3.reset_index(drop=True)

location_df_4["Obesity Rates"] = obesity_rates.tolist()

location_df_4

# Configure gmaps with API key
gmaps.configure(api_key=g_key)

# Store 'Lat' and 'Lng' into  locations 
locations_heat = location_df_4[["Lat", "Lng"]].astype(float)

# Convert Obesity Rates to float and store
obesity_rates_heat_map = obesity_rates

# Create a Heatmap for Obesity Rates layer
fig = gmaps.figure()

heat_layer = gmaps.heatmap_layer(locations_heat, weights=obesity_rates, 
                                 dissipating=False, max_intensity=obesity_rates.max(),
                                 point_radius = 1)

fig.add_layer(heat_layer)

fig

# File to Load
file_to_load_2 = "Resources/Covid_Deaths.csv"

# Read Obesity CSV and store into Pandas data frame
covid_deaths_full = pd.read_csv(file_to_load_2)

# Split Deaths from State
new_column_2 = covid_deaths_full["Covid Deaths"].str.split(":", n = 1, expand = True) 
covid_deaths_full["State Name"]= new_column_2[0] 
covid_deaths_full["Deaths"]= new_column_2[1] 

# Only Keep State and Deaths Columns
covid_deaths_reduced = covid_deaths_full.loc[:, ["State", "Deaths"]]

# Sort alphabetically and reset index
covid_deaths_reduced.sort_values(by=['State', 'Deaths'], ascending=[True, False])
covid_deaths_sorted = covid_deaths_reduced.reset_index(drop=True)

# Merge Obesity Data with COVID Data
combined_data_df = pd.merge(location_df_4, covid_deaths_sorted, how='outer', on='State')

# Drop nan values
combined_clean = combined_data_df.dropna()
combined_clean

# Convert covid_deaths to float
covid_deaths = combined_clean.iloc[:,4].astype(float)

# Check covid deaths data type
covid_deaths.dtypes

# Correlation Analysis between obesity rates and covid deaths per state
obesity = combined_clean.iloc[:,3]
covid_deaths = combined_clean.iloc[:,4].astype(float)
correlation = st.pearsonr(obesity,covid_deaths)
print(f"The correlation coefficient between 2019 obesity and covid deaths in the United States is {round(correlation[0],2)}")

# Regression Analysis between obesity rates and covid deaths per state
x_values = obesity
y_values = covid_deaths
(slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)
regress_values = x_values * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
plt.scatter(x_values,y_values)
plt.plot(x_values,regress_values,"r-")
plt.annotate(line_eq,(6,10),fontsize=15,color="red")
plt.title('Obesity Rates v. Covid Deaths')
plt.xlabel('Obesity')
plt.ylabel('Covid Deaths')
plt.show()
plt.savefig("output_data/Fig12.png")

# Configure gmaps with API key
gmaps.configure(api_key=g_key)

# Store 'Lat' and 'Lng' into  locations 
locations_covid_heat = combined_clean[["Lat", "Lng"]].astype(float)

# Convert Obesity Rates to float and store
covid_deaths_heat = combined_clean.iloc[:,4].astype(float)

# Create a Heatmap for Covid Deaths layer
fig = gmaps.figure()

heat_layer = gmaps.heatmap_layer(locations_covid_heat, weights=covid_deaths_heat, 
                                 dissipating=False, max_intensity=covid_deaths_heat.max(),
                                 point_radius = 1)

fig.add_layer(heat_layer)

fig
