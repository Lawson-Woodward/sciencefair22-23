import matplotlib.pyplot as plt
import numpy as np

xPoints = np.array([0,1,2,5,6,8,9]) # If not used, the x axis defaults to 0, 1, 2, ...
yPoints = np.array([-1,4,5,4,4,3,-30])
plt.title("My mental health")

# If you don't include "marker = ", a scatterplot will be shown
plt.plot(xPoints, yPoints, marker = '1')
#plt.show()

"""
Marker	Description 
'o'	    Circle	
'*'	    Star	
'.'	    Point	
','	    Pixel	
'x'	    X	
'X'	    X (filled)	
'+'	    Plus	
'P'	    Plus (filled)	
's'	    Square	
'D'	    Diamond	
'd'	    Diamond (thin)	
'p'	    Pentagon	
'H'	    Hexagon	
'h'	    Hexagon	
'v'	    Triangle Down	
'^'	    Triangle Up	
'<'	    Triangle Left	
'>'	    Triangle Right	
'1'	    Tri Down	
'2'	    Tri Up	
'3'	    Tri Left	
'4'	    Tri Right	
'|'	    Vline	
'_'	    Hline

"""

plt.plot(xPoints, yPoints, 'o:r') # 'o : r' = 'marker :line: color'
#plt.show()

""""
If you leave this argument blank for 'plt.plot()', no line will be graphed only the points

Line Syntax	    Description
'-'	            Solid line	
':'	            Dotted line	
'--'	        Dashed line	
'-.'	        Dashed/dotted line

Keep in mind that you can also use hex values for colors

Color 	Description
'r'      Red	
'g'	    Green	
'b'	    Blue	
'c'	    Cyan	
'm'	    Magenta	
'y'	    Yellow	
'k'	    Black	
'w'	    White
#4CAF50	Hex Shade of Green

* hex codes can not be used to color the line of a plot/graph

For more color names:   
https://www.w3schools.com/colors/colors_names.asp  
"""
    
# 'ms' stands for "marker size"
plt.plot(xPoints, yPoints, 'b-x', ms = 15)
#plt.show()

# 'mec' stands for "marker edge color", which uses the same color syntax listed above 
plt.plot(xPoints, yPoints, 'b-o', ms = 15, mec = 'r')
#plt.show()

# 'mfc' stands for "marker face color" - not the same as the line color, which uses the same color syntax listed above 
plt.plot(xPoints, yPoints, 'b-o', ms = 15, mec = 'r', mfc = '#FF00FF')
#plt.show()

# 'ls' stands for "linestyle", which takes precedence over 'b--o' (if you look in the terminal while running this code, it will also inform you of the redundancy)
plt.plot(xPoints, yPoints, 'b--o', ms = 15, mec = 'r', mfc = '#FF00FF', ls = 'dotted')
#plt.show()

# 'c' stands for "color" (of the line), which takes precedence over 'b--o'
plt.plot(xPoints, yPoints, 'b--o', ms = 15, mec = 'r', mfc = '#FF00FF', ls = 'dotted', c = '#7C33CF')
#plt.show()

# lw stands for "line width "
plt.plot(xPoints, yPoints, 'b--o', ms = 15, mec = 'r', mfc = '#FF00FF', ls = '-', c = '#7C33CF', lw = 10.34)
#plt.show()

# You can also have multiple x and y lists to graph more than one function at once
# With xlabel() and ylabel() you can name the x and y axes and with title() you can add a title to the graph 
# loc() changes the position text (legal values are 'left', 'right', and 'center')
# This also means that text can be edited:
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# You have to create fonts before implementing them
font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)
plt.plot(x, y)
plt.grid(axis = 'x', c = 'green', ls = '-.', lw = 1) # creates a grid on the graph
#plt.show()

# The subplot() function allows you display multiple graphs in one figure
# Arguments: (rows, columns, nth plot) <- this describes the format of the FIGURE, not the graph
x1Points = [0,1,4,9,16,25,36]
y1Points = [0,1,2,3,4,5,6]
plt.subplot(2,1,1)
plt.plot(x1Points, y1Points)
plt.title('Graph')

x2Points = [0,1,2,3,4,5,6]
y2Points = [0,1,4,9,16,25,36]
plt.subplot(2,1,2) # because this is the 2nd graph
plt.plot(x2Points, y2Points)
plt.title('Inverse', loc = 'center')

plt.suptitle("Two Graphs") #supertitle of all the figures
plt.show()

# The scatter() function allos you to draw a scatterplot of data 
# Arguments: Two arrays of equal length
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
    # "colors" must have the same length as the x and y arrays (one color for each point)
colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
plt.scatter(x, y, color = colors) # you could also write " c = '#88c999' "
plt.xlabel("Age of Car")
plt.ylabel("Speed of Car")
plt.suptitle("I Like Cars I Think (✿◡‿◡)")
#plt.show()
plt.close() # This  simply just refreshes the figure so multiple graphs don't overlap

# You can also use built-in color ranges:
y = np.array([5,7,8,7,2,17,2,9,4,11])
x = np.array([99,86,87,88,111,86,103,87,94,78])
colors = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
sizes = np.array([10, 20, 120, 400, 50, 60, 70, 133, 90, 600])
plt.scatter(x, y, c = colors, cmap = 'viridis', s = sizes, alpha = 0.2)
    # the cmap is like the 'theme' of colors, and we pick those specific colors of the them with our 'colors' array
    # alpha is the transparency level
plt.colorbar() # this includes the colorbar on the side
#plt.show()
plt.close()
''' 
AVAILABLE COLOR MAPS:
https://www.w3schools.com/python/matplotlib_scatter.asp 
'''
# You can also create bar graphs with matplotlib using the bar() method
x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])
plt.bar(x,y, color = 'red', width = 0.2) # 'barh()' if you want horizontal bars, but makre sure to change the height, NOT width
#plt.show()
plt.close()

# You can also use the hist() function to create histograms in matplotlib, which helps represent frequency distributions 
x = np.random.normal(170, 10, 250)
    # values concentrate around 170, standard deviation of 0.10, 250 total values
plt.hist(x)
plt.title("Frequency Distributions")
#plt.show()
plt.close()

# The pie() function can be used to draw pie charts in matplotlib
values =np.array ([1,2,3,4])
    # By default, the first wedge starts from the right on the x axis and moves COUNTERCLOCKWISE
    # However, you can change this with the 'startangle' argument
myLabels = np.array(['Option 1', 'Option 2', 'Option 3', 'Option 4'])
explodeValues = np.array([0,0,0.4, 0])
    # 'explode' is used to make certain wedges stand out (one explode value per wedge)
wedgeColors = np.array(["black", "hotpink", "b", "#4CAF50"])
plt.pie(values, labels = myLabels, startangle = 35, explode = explodeValues, shadow = True, colors = wedgeColors)
plt.legend(title = 'Four Options: ', bbox_to_anchor=(1.31,0.5), loc = 'right')
    # 'bbox_to_anchor' allows us to choose where the legend will be places (expands down and to the right from a corner point) relative to the 'loc'
plt.show()

# If we encounter more issues with overlapping legends, an in-depth explanation can be found here:
# https://stackoverflow.com/questions/43272206/python-legend-overlaps-with-the-pie-chart 