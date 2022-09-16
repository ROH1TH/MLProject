IE7374 FINAL PROJECT REPORT
Room Occupancy Prediction using Multiclass Classification models
Made by

Bhavya Budda - budda.b@northeastern.edu
Jayanth Reddy Rachamallu- rachamallu.j@northeastern.edu
Prachi Patel - patel.prachi2@northeastern.edu
Rohith Kanakagiri - kanakagiri.r@northeastern.edu
Problem Setting & Definition:
Building emissions account for almost 28% of the world’s total greenhouse gas emissions. With the growing need of better regulating these emissions, the use of sensor technology to predict the number of occupants in the room and automatically manage the operations of a building to better suit the number of occupants would help the building operate more efficiently and reduce the emissions caused due to building operation. Apart from the environmental benefits of determining the number of occupants in the room, it has benefits on a more global scale as well. With the increase in covid-19 cases the world has seen in the past couple of years, small gatherings in closed spaces turned out to be the one of the key factors that fostered the exponential rise and spread of the virus. Predicting the occupancy and reducing the number of small gatherings will help curb the spread of any such future pandemics.

The goal of this project is to build a model to predict the number of occupants in a room based on data from light, sound, CO2 , temperature, and motion sensors fit in a 6m * 4.6m room.The sensor data was collected over the duration of 4 days. With this project, we try to classify the number of occupants into 4 pools (0,1,2 and 3) using Machine Learning techniques and algorithms.

Data Sources:
UCI Repository link

Data Description:
The dataset has 17 attributes, 10,129 instances, and 1 outcome variable. The 17 attributes are: Time: The time of the day at which the sensor measurements were taken in HH:MM:SS format S1-4 Temp: The temperature measured in degree Celsius at 4 different locations of the room S1-4 Light: The amount of light measured in Lux at 4 different locations of the room S1-4 Sound: The sound emitted measure in Volts at 4 different locations of the room S5_CO2: The amount of CO2 in the room in Parts per Million (PPM) S5_CO2_Slope: The slope of CO2 values taken in the room at a sliding window S6,7 PIR: Motion detection sensor in two different parts of the room.

Exploratory Data Analysis and Visualization:
