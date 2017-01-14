import os, sys
print(sys.argv[1])
infile = open(sys.argv[1], encoding="utf8")
data = infile.read()
infile.close()
outfile = open(sys.argv[1], "w", encoding="utf8")

data = data.replace("%matplotlib notebook", "#%matplotlib notebook")
data = data.replace("%matplotlib inline #reset", "#%matplotlib inline #reset")
data = data.replace("time.sleep", "#time.sleep")
data = data.replace("fig.canvas.draw", "#fig.canvas.draw")
data = data.replace("plt.gcf().canvas.draw", "#plt.gcf().canvas.draw")

outfile.write(data)
outfile.close()
