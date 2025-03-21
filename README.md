# name_generator
A machine learning model to generate new Finnish surnames

When my husband and I were getting married, we wanted to choose a completely unique surname — one that no one else in Finland had.

Finnish name law requires that new family names must sound Finnish and be recognizable as names. To find a suitable surname, we decided to build a machine learning model to generate potential options that met these criteria.

For training, we used open-source data from the Digital and Population Data Services Agency, which provided information on existing Finnish surnames. The model was fed a string of names and then predicted the most probable sequences of characters based on the input data.

We quickly noticed that the quality of the generated names heavily depended on the input data. To refine our results, we experimented with different datasets, including Swedish surnames and Finnish place names (such as cities, towns, lakes, and other geographical locations, which often resemble Finnish surnames).

In the end, we were quite happy with our model and the names it generated—but we ultimately chose a surname from my mother’s side! 😊
