# Use RANSAC regression to detect outliers


# Robust methods are preferable when identifying outliers.
# Working with median and median absolute deviation cannot tell the difference between continuous shifts in values and true outliers.

# is robust against outliers.
# can tell the difference between continuous shifts in values and outliers.

import seaborn as sns
sns.regplot(x=list(range(60)), y=df_temp4.iloc[[144], :-1].astype(float))

from sklearn.linear_model import RANSACRegressor

# RANSAC regression with the parameter residual_threshold=1. 
# It specifies the maximal residual (difference between the predicted value from the actual observed value) for a data sample to be classified as an inlier (non-outlier).
model_ransac = RANSACRegressor(residual_threshold=1)
features = pd.DataFrame({'x': range(60)})
target = df_temp4.iloc[144, :-1]

model_ransac.fit(features, target)

# It shows which data points do not count as outliers, what we call inliers.
model_ransac.inlier_mask_

# initialise figure and axes
fig, ax = plt.subplots()

# draw scatter plot
sns.scatterplot(x=list(range(60)),
                y=df_temp4.iloc[144, :-1],
                hue=model_ransac.inlier_mask_, #coloring inliers (orange) and outliers (blue)
                ax=ax)

# optimise plot
ax.set(xlabel='Time since cycle start [sec]',
       ylabel='Temperature [°C]',
      title='Temperature sensor 4 data of hydraulic pump')
ax.legend(title='Inlier')

# Using RANSAC regression to identify outliers
def outlierdetection(df_temp):
    """Detecting outliers for temperature DataFrames. Only rows with outliers will be returned. Inliers will be masked with NaN
    df_temp = placeholder for read temperature sensor DataFrame"""

    # copy DataFrame for structure
    temp_values = df_temp.copy().set_index('cycle_id')
    df_outliers_mask = temp_values.copy()

    # instantiate RANSAC regression model with adjusted outlier threshold
    model_ransac = RANSACRegressor(residual_threshold=1)

    for cycle in range(len(df_temp)):  # for each cycle
        # target vector for this cycle
        features = pd.DataFrame({'x': range(60)})
        target = temp_values.iloc[cycle, :]

        # fit model with data of this cycle
        model_ransac.fit(features, target)

        # save outlier mask of this cycle
        df_outliers_mask.iloc[cycle, :] = ~model_ransac.inlier_mask_

    # filter only outlier values
    outlier_values = temp_values[df_outliers_mask]

    # return only rows that contain outliers
    output = outlier_values[~outlier_values.isna().all(axis=1)]
    return output

# Remember:

# Outliers are false in model_ransac.inlier_mask_.
# You can create multiple sheets in an Excel workbook with pd.ExcelWriter() and pd.to_excel().
