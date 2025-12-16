from Sub_Functions.Read_data import Read_data
from Sub_Functions.Analysis import Analysis
from Sub_Functions.Read_data import build_classification_head
from Sub_Functions.Plot import ALL_GRAPH_PLOT
from Sub_Functions.Plot2 import ALL_GRAPH_PLOT2
from Sub_Functions.Evaluate import open_popup

Choose = open_popup("Do you need Complete Execution:?")

if Choose == "Yes":

    DB = ["Rainfall_Dataset", "Temperature_Dataset", "Wind_Dataset"]
    # DB = ["Temperature_Dataset", "Wind_Dataset"]

    # for i in range(len(DB)):
    #     Read_data(DB[i])
    #
    # build_classification_head()

    TP = Analysis(DB[1])

    # TP.COMP_Analysis()

    TP.PERF_Analysis()

    DB1 = ["Rainfall_Dataset", "Temperature_Dataset", "Wind_Dataset", "class_model"]

    for j in range(len(DB1)):

        if DB1[j] == "class_model":
            plot1 = ALL_GRAPH_PLOT2()

            plot1.GRAPH_RESULT2(DB1[j])
        else:

            plot2 = ALL_GRAPH_PLOT()

            plot2.GRAPH_RESULT(DB1[j])
else:

    DB1 = ["Rainfall_Dataset", "Temperature_Dataset", "Wind_Dataset", "class_model"]
    # DB1 = ["Wind_Dataset"]

    for j in range(len(DB1)):

        if DB1[j] == "class_model":
            plot1 = ALL_GRAPH_PLOT2()

            plot1.GRAPH_RESULT2(DB1[j])
        else:

            plot2 = ALL_GRAPH_PLOT()

            plot2.GRAPH_RESULT(DB1[j])

