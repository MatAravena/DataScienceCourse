# You should always understand outliers as well as you can, because they directly influence the learning success of your machine learning models. 
# They can even render the predictions of machine learning models completely useless. 
# The reason for this is that a machine learning model looks for a universal structure in the data and outliers create a false idea of this structure. 
# So how you deal with outliers is a key point for every data science project. The origin of the outliers determines how you should handle them. 
# It is very important not to consider outliers as negative, but as a normal property of real data sets. 
# They have their own structure and therefore potentially contain a lot of information about important characteristics of a data set.



import sqlalchemy as sa
engine = sa.create_engine('sqlite:///hydro.db')
connection = engine.connect()

sa.inspect(engine).get_table_names()


fig, axs = plt.subplots(nrows=4,
                       figsize=[6, 12])

for sensor in range(1, 5):  # for each thermometer
    
    # read in data from database
    sql_query = '''
    SELECT *
    FROM temperature_sensor_{}
    '''.format(sensor)
    df = pd.read_sql(sql_query, con=connection)
    
    # create line chart
    df.iloc[:, :-1].T.plot(legend=False, ax=axs[sensor - 1])

    # optimise line chart
    axs[sensor - 1].set(xlabel='Time since cycle start [sec]',
                        ylabel='Temperature [°C]',
                        title='Temperature sensor {} data of hydraulic pump'.format(sensor))
     
    axs[sensor - 1].set_xticklabels(range(-10, 71, 10));

fig.tight_layout()  # avoid overlapping axes text


# Visualizing outliers in histograms
# Checking the value in 1 row which is the temperatures un 1 hour and checking outliers as Temp outliers
fig, axs = plt.subplots(nrows=4,
                        figsize=[6, 12], 
                        sharey=True)

for sensor in range(1, 5):  # for each thermometer
    
    # read in data from database
    sql_query = '''
    SELECT *
    FROM temperature_sensor_{}
    '''.format(sensor)
    df = pd.read_sql(sql_query, con=connection)
    
    # create histogram for cycle 145
    df.iloc[144, :-1].plot(kind='hist',
                           ax=axs[sensor - 1])
    
    # optimise histogramm
    axs[sensor - 1].set(xlabel='Temperature [°C]',
                        ylabel='Count',
                        title='Temperature sensor {} data of hydraulic pump'.format(sensor))

fig.tight_layout()  # avoid overlapping axes text


# Remember:
# An outlier is a data point that differs greatly from the majority of all other data points.
# Histograms are particularly well suited for visualizing outliers.
# Outliers can result from faulty data acquisition or storage, but they can also represent valid data points.