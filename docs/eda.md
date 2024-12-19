Conducting explorative analysis on the numerical data. Mainly focuses on looking at changes in study habits reflected in study times.

## Temporal Analysis:
Looking at some overall temporal trend and how study hours flunctuate in different week days, different quarters, and different years. Looks like study time overall increases.

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../demos/fall24/trend_all.png" style="width:60%; height:auto;">
     <img src="../demos/fall24/sliding_window.png" style="width:40%; height:auto;">
</div>

We can also look at some specific categories of what I do, specifcally speaking (`research`, `dsc`, `math`, and `cogs`), we cna spot some category specific trend.

<div style="text-align: center;">
    <img src="../demos/fall24/trend_cat.png" style="width:70%; height:auto;">
</div>

We can also examine how study hours changes as a function of ***week days***, ***week number***, and ***year***.

<div style="text-align: center;">
    <img src="../demos/fall24/heatmap_weekdays.png" style="width:100%; height:auto;">
</div>

We can also look at season's effect on study times by a heatmap againwhere we can see that there is an overall increase in study time over the year and the most study time at ***Fall*** and ***Spring*** quarter.

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../demos/fall24/heatmap_quarters.png" style="width:40%; height:auto;">
</div>

## Dimensionality Reductions
Doing some dimensionality reduction technique to show some underlaying property of each quarter being different, specifcally showing different study habits.
<div style="display: flex; justify-content: center; align-items: center;">
    <img src="../demos/fall24/pca.png" style="width:50%; height:auto;">
    <img src="../demos/fall24/tsne.png" style="width:50%; height:auto;">
</div>


## Categorical Course Analysis
Created an overall statistics of how each course varies on time usage, having details relative to each classes.

<div style="text-align: center;">
    <img src="../demos/fall24/all_cat.png" style="width:90%; height:auto;">
</div>
