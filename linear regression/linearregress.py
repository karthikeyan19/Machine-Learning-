from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import random
style.use('ggplot')
#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_data_set (hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64),np.array(ys, dtype=np.float64)
xs, ys = create_data_set(40,40,2,correlation='pos')
def best_fit_slope_and_interception (xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    b = mean(ys) - m * mean(xs)
    return m, b

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig,ys_line)
    squared_error_y_mean = squared_error(ys_orig,y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

m, b = best_fit_slope_and_interception (xs,ys)
plot.scatter(xs, ys)
regression_line = []
for x in xs:
    regression_line.append((m*x)+b)
r_squared = coefficient_of_determination(ys,regression_line)
print(r_squared)

plot.plot(xs, regression_line)
predict_x = 8
predict_y = m * x + b
#plot.scatter(predict_x, predict_y ,color = 'g')

plot.show()
