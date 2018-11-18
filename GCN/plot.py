import ggplot as gp
import pandas as pd
x=np.array(range(len(acculist)))
acculist=np.sort(acculist)
import matplotlib.pyplot as plt
plt.plot(x,acculist, linewidth=1.0)
plt.xlabel('legislation',fontsize=20)
plt.ylabel('the rate of vote yea',fontsize=20)
plt.title('the percentage of the yea vote',fontsize=20)
plt.grid(True)
plt.savefig("rate.png",fontsize=20)
plt.show()

plt.close()
labels = 'Yea', 'Not_vote', 'Nay'
sizes = votenumber
colors = ['lightblue', 'gold', 'lightcoral']
explode = (0.1, 0, 0) # only "explode" the 2nd slice (i.e. 'Hogs')
plt.savefig('pie.png')
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
acculist=cosponsors_number
x=np.array(range(len(acculist)))
acculist=np.sort(acculist)
import matplotlib.pyplot as plt
plt.plot(x,acculist, linewidth=1.0)
plt.xlabel('legislation',fontsize=25)
plt.ylabel('the number of cosponsors',fontsize=25)
plt.title('the number of cosponsors for every legislation',fontsize=25)
plt.grid(True)
plt.savefig("cosponsor.png",fontsize=20)
plt.show()
import matplotlib.pyplot as plt
plt.close()