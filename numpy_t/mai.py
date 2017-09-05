import  matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-10,10,50)

y=2*x**3+4*x+3
y2=33*x


plt.figure(num=3,figsize=(8,5))
l1,=plt.plot(x,y2,label='up')
l2,=plt.plot(x,y,color='red',linewidth=1,linestyle='--',label='down')

#註解線段
x0=0.5
y0=2*x0**3+4*x0+3
plt.scatter(x0,y0,s=50,color='b')
plt.plot([x0,x0],[y0,0],'k--',lw=1.0)
plt.annotate('co line', xy=(x0, y0), xytext=(3, -1000),
            arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=0.2'),
            )
#註解文字
plt.text(-3.7,1114,'coword',fontdict={'size':16,'color':'r'})

#對線命名
plt.legend(handles=[l1,l2],labels=['wdds','rere'],loc='best')

plt.xlabel('x')
new_ticks=np.linspace(-1,20,5)
print(new_ticks)

#y軸編號
#plt.yticks([-2,1.6],['good','bad'])
#plt.xticks(new_ticks)


#去掉邊框
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#把軸心固定在x=0,y=0
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))



plt.show()