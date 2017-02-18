##import random
##arr=["neo","leo","tom","jerry"]
####print(random.choice(arr))
##print(arr)
##random.shuffle(arr)
##print(arr)
##import sys
##import time
###sys.exit()
####v=sys.stdin.readline()
#####print(v)
####sys.stdout.write(v)
####print()
####print(sys.version)
####print(time.localtime())
##for x in range(1,10):
##    print(x)
##    time.sleep(1)

##import turtle
##t=turtle.Pen()
##t.reset()
####for x in range(1,38):
####    t.forward(100)
####    t.left(175)
##t.circle(50)
##t.end_fill()
##def hello():
##    print("hello world")
##from tkinter import *
##tk=Tk()
##btn=Button(tk,text="click me",command=hello)
##btn.pack()

##from tkinter import *
##tk=Tk()
##canvas=Canvas(tk,width=500,height=500)
##canvas.pack()
##canvas.create_line(0,0,500,500)
##ra=canvas.create_rectangle(10,10,300,300)
##canvas.create_arc(10,10,200,100,extent=180,style=ARC)
##canvas.create_text(150,100,text="this is my canvas ",font=('微软雅黑',50))
##canvas.itemconfig(ra,fill='blue')
##img=PhotoImage(file="C:\\python_code\\x.png")
##canvas.create_image(0,0,anchor=NW,image=img)

from tkinter import *
import random
import time

class Ball:
    def __init__(self,canvas,color):
        self.canvas=canvas
        self.id=canvas.create_oval(10,10,25,25,fill=color)
        self.canvas.move(self.id,245,100)
        starts=[-3,-2,-1,1,2,3]
        random.shuffle(starts)
        self.x=-1
        self.y=-6
        self.canvas_height=self.canvas.winfo_height()
        self.canvas_width=self.canvas.winfo_width()
    def draw(self):
        self.canvas.move(self.id,self.x,self.y)
        pos=self.canvas.coords(self.id)
        if pos[1]<=0:
            self.y=3
        if pos[3]>=self.canvas_height:
            self.y=-3
        if pos[0]<=0:
            self.x=3
        if pos[2]>=self.canvas_width:
            self.x=-3
    
tk=Tk()
tk.title("Game")
tk.resizable(0,0)
tk.wm_attributes("-topmost",1)
canvas=Canvas(tk,width=500,height=400,bd=0,highlightthickness=0)
canvas.pack()
tk.update()
ball=Ball(canvas,'red')
while 1:
    ball.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.01)


