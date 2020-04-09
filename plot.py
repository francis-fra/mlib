#============================================================
# plotting
#============================================================
def plot_top_counts(data, ishor=False):
    keys = [item[0] for item in data]
    values = [item[1] for item in data]
    if ishor == False:
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.bar(x='categories', y='counts', rot=0)
    else:
        keys.reverse()
        values.reverse()
        df = pd.DataFrame({'categories': keys, 'counts': values})
        ax = df.plot.barh(x='categories', y='counts', rot=0)

def plot_series(s, ishor=False):
    "bar chart for series"
    if ishor == True:
        s = s.sort_values(ascending=True)
        df = pd.DataFrame(s).plot.barh()
    else:
        df = pd.DataFrame(s).plot.bar()

