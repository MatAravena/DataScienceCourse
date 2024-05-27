# PDPs and ICE Plots
# two new agnostic models

# Investigated the average influence of a feature on the predictions.
# Visualized the influence of the values of a feature for each data point.
# Investigated the shared average influence of two feature on the predictions.



# Partial dependence plots
# only gives us the importance of a feature and does not describe how the values the feature takes on affect our predictions.

# two more methods in this lesson: *partial dependence plots* (PDPs) and *individual conditional expectation plots* (ICE plots).
#  they can be used for all models like random forest, ANNs or SVM

# PDPs are a global method because they show the average effect of a feature value on the prediction.

# read data
import pandas as pd
df_train = pd.read_csv('attrition_train.csv')

# split training data into features and target
features_train = df_train.iloc[:, 1:]
target_train = df_train.loc[:, 'attrition']

df_train.head()

# Import, instantiate and fit random forest classifier
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=12, random_state=0)
model_rf.fit(features_train, target_train)


# sns.stripplot(x=str,                 # Data on the x-axis
#               y=str,                 # Data on the y-axis
#               data=pd.DataFrame,     # Underlying DataFrame
#               jitter=float           # Spread of overlapping points  
#               alpha=float)           # Transparency of the points in the plot

# However, we weren't able to say how 'monthlyincome' affects the predictions, i.e. whether a low income has a greater impact than a high income, 

import seaborn as sns
sns.stripplot(x='attrition', y='monthlyincome', data=df_train, alpha=0.5, jitter=1)

# PDPs show the marginal influence a feature has on an average prediction. 
# They can visually represent both linear and non-linear relationships between features and predictions.

# To calculate the partial dependency of a selected feature, the value of this feature is replaced by a given value in all data points. 
# Then the average of all predictions is calculated before the next value is tried out.

# are needed 2 functions
from pdpbox  import pdp

# pdp.pdp_isolate(model=model,          # a fitted sklearn model
#                 dataset=DataFrame,    # the dataset on which the model was trained
#                 model_features=list,  # names of all the features the model uses
#                 feature=`str`         # feature's column name in `dataset`
#                )

# calculate the partial dependency
pdp_monthlyincome = pdp.pdp_isolate(model=model_rf,                         # a fitted sklearn model
                                    dataset=features_train,                 # the dataset on which the model was trained
                                    model_features=features_train.columns,  # names of all the features the model uses
                                    feature='monthlyincome'                 # feature's column name in `dataset`
                                   )

# pdp.pdp_plot(pdp_isolate_out=instance of PDPIsolate, # output of pdp.isolate()
#              feature_name=str,                        # name of feature (for title)
#              center=bool,                             # center the plot
#              plot_pts_dist=bool                       # display real feature values
#             )

#  We can visualize it with the second function pdp.pdp_plot()
pdp.pdp_plot(pdp_isolate_out=pdp_monthlyincome, # output of pdp.isolate()
             feature_name='Monthly Income',     # name of feature (for title)
             center=False,                      # center the plot
             plot_pts_dist=True                 # display real feature values
            )
# The light blue sleeve shows the standard deviation of the predictions. The blue points represent the values which the partial dependency was actually calculated for. 



# # Individual conditional expectation plot
# PDPs are a global method because they show the average effect of a feature value on the prediction.
# The local version of a PDP is the individual conditional expectation (ICE) plot. 


pdp.pdp_plot(pdp_isolate_out=pdp_monthlyincome, 
             feature_name="Monthly Income",  
             plot_lines=True, 
             plot_pts_dist=True,
             center=False);


# Center the point from the origin of the feature
fig, ax_dict=pdp.pdp_plot(pdp_isolate_out=pdp_monthlyincome,
             feature_name="Monthly Income",
             plot_lines=True,
             plot_pts_dist=True,
             center=True)
ax_dict['pdp_ax']['_pdp_ax'].set_ylim(-0.5,0.5);



# Apprently the are suggestions to change the values and reordenate the accuracy of the predictions
features_aim = pd.read_csv('employee_single.csv', header=None, index_col=0).T
features_aim

model_rf.predict_proba(features_aim)
# array([[0.16173827, 0.83826173]])     <------------------


# HR asked this to test with this new values
# A salary increase of up to 20%
# A promotion accompanied by a salary increase of 10%
# Hire an additional person in the department so that they no longer need to do so much overtime

# First scenario:
features_aim_1 = features_aim.copy()
features_aim_1.loc[:, 'monthlyincome'] = features_aim_1.loc[:, 'monthlyincome']  * 1.2
print('20% raise improves chance of not leaving by:', model_rf.predict_proba(features_aim)-model_rf.predict_proba(features_aim_1))
# 20% raise improves chance of not leaving by: [[-0.1684527  0.1684527]]                                    <------------------

# Second scenario:
features_aim_2 = features_aim.copy()
features_aim_2.loc[:, 'monthlyincome'] = features_aim_2.loc[:, 'monthlyincome']  * 1.1
features_aim_2.loc[:, 'joblevel'] = features_aim_2.loc[:, 'joblevel']  + 1
print('10% raise with promotion improves chance of not leaving by:', model_rf.predict_proba(features_aim)-model_rf.predict_proba(features_aim_2))
# 10% raise with promotion improves chance of not leaving by: [[-0.3065472  0.3065472]]                     <------------------

# Third scenario:
features_aim_3 = features_aim.copy()
features_aim_3.loc[:, 'overtime'] = 0
print('reducing overtime improves chance of not leaving by:', model_rf.predict_proba(features_aim)-model_rf.predict_proba(features_aim_3))
# reducing overtime improves chance of not leaving by: [[-0.41098546  0.41098546]]                          <------------------

# An increase in salary by itself reduces the probability by about 17%. The small salary increase as part of a promotion makes a difference of about 31%. 
# However, according to our model, hiring another colleague to relieve this person would be the best option with a 45% reduction.




# # PDPs with two features
# One disadvantage of PDPs and ICE plots is that they don't take into account the interaction between the features.

pdp_income_joblevel = pdp.pdp_interact(model=model_rf, 
                                           dataset=features_train, 
                                           model_features=features_train.columns, 
                                           features=['monthlyincome', 'joblevel'])

# pdp.pdp_interact_plot(pdp_interact_out=pdp_interact_out,  # output of pdp.pdp_interact()
#                  feature_names=list,                     # name of the feature of interest (for plot title)
#                  plot_type=str                           # type of interaction plot
#                  )


pdp.pdp_interact_plot(pdp_interact_out=pdp_income_joblevel,    # output of pdp.pdp_interact()
                 feature_names=['Monthly Income', 'Joblevel'], # name of the feature of interest (for plot title)
                 plot_type='grid'                              # type of interaction plot
                 );
# Always remember that a model does not perfectly reflect reality. It can only help us to better understand reality.

# Remember:

# Model agnostic methods help us interpret black-box models.
# Partial dependence plots (PDPs) show the average influence of feature values on the predictions
# Independent conditional expectation plots (ICE plots) visualize the influence of feature values for each data point separately
# PDP plots show the average of the ICE plots
# You can create PDPs and ICE plots with the pdpbox module.