from tkinter import *
from tkinter.messagebox import *
import random
import time
#俄罗斯方块界面的高度
HEIGHT  = 18
#俄罗斯方块界面的宽度
WIDTH   = 10
ACTIVE  = 1
PASSIVE = 0
TRUE    = 1
FALSE   = 0
root=Tk();root.title('Russia')
class App(Frame):
    def __init__(self,master):
        Frame.__init__(self)
        master.bind('<Up>',self.Up)
        master.bind('<Left>',self.Left)
        master.bind('<Right>',self.Right)
        master.bind('<Down>',self.Down)
        #master.bind('<Down>',self.Space)
        master.bind('<space>',self.Space)
        master.bind('<Control-Shift-Key-F12>',self.Play)
        master.bind('<Key-F6>',self.Pause)
        self.backg="#%02x%02x%02x" % (120,150,30)
        self.frontg="#%02x%02x%02x" % (40,120,150)
        self.nextg="#%02x%02x%02x" % (150,100,100)
        self.flashg="#%02x%02x%02x" % (210,130,100)
        self.LineDisplay=Label(master,text='Lines: ',bg='black',fg='red')
        self.Line=Label(master,text='0',bg='black',fg='red')
        self.ScoreDisplay=Label(master,text='Score: ',bg='black',fg='red')
        self.Score=Label(master,text='0',bg='black',fg='red')
        #Display time
        self.SpendTimeDisplay=Label(master,text='Time: ',bg='black',fg='red')
        self.SpendTime=Label(master,text='0.0',bg='black',fg='red')
        self.LineDisplay.grid(row=HEIGHT-2,column=WIDTH,columnspan=2)
        self.Line.grid(row=HEIGHT-2,column=WIDTH+2,columnspan=3)
        self.ScoreDisplay.grid(row=HEIGHT-1,column=WIDTH,columnspan=2)
        self.Score.grid(row=HEIGHT-1,column=WIDTH+2,columnspan=3)
        #Display time
        self.SpendTimeDisplay.grid(row=HEIGHT-4,column=WIDTH,columnspan=2)
        self.SpendTime.grid(row=HEIGHT-4,column=WIDTH+2,columnspan=3)
        self.TotalTime=0.0
        self.TotalLine=0;self.TotalScore=0
        #Game over
        self.isgameover=FALSE
        #Pause
        self.isPause=FALSE
        #Start
        self.isStart=FALSE
        self.NextList=[];self.NextRowList=[]
        r=0;c=0
        for k in range(4*4):
            LN=Label(master,text='    ',bg=str(self.nextg),fg='white',relief=FLAT,bd=4)
            LN.grid(row=r,column=WIDTH+c,sticky=N+E+S+W)
            self.NextRowList.append(LN)
            c=c+1
            if c>=4:
                r=r+1;c=0
                self.NextList.append(self.NextRowList)
                self.NextRowList=[]
        self.BlockList=[];self.LabelList=[]
        self.BlockRowList=[];self.LabelRowList=[]
        row=0;col=0
        for i in range(HEIGHT*WIDTH):
            L=Label(master,text='    ',bg=str(self.backg),fg='white',relief=FLAT,bd=4)
            L.grid(row=row,column=col,sticky=N+E+S+W)
            L.row=row;L.col=col;L.isactive=PASSIVE
            self.BlockRowList.append(0);self.LabelRowList.append(L)
            col=col+1
            if col>=WIDTH:
                row=row+1;col=0
                self.BlockList.append(self.BlockRowList)
                self.LabelList.append(self.LabelRowList)
                self.BlockRowList=[];self.LabelRowList=[]
        #file
        fw=open('text.txt','a')
        fw.close()
        hasHead=FALSE
        f=open('text.txt','r')
        if f.read(5)=='score':
            hasHead=TRUE
        f.close()
        self.file=open('text.txt','r+a')
        if hasHead==FALSE:
            self.file.write('score    line    time    scorePtime    linePtime    scorePline    date/n')
            self.file.flush()
            
        self.time=1000
        self.OnTimer()
    def __del__(self):
        #self.file.close()
        pass
        
    def Pause(self,event):
        self.isPause=1-self.isPause
    def Up(self,event):
        BL=self.BlockList;LL=self.LabelList
        Moveable=TRUE
        xtotal=0;ytotal=0;count=0
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if LL[i][j].isactive==ACTIVE:
                    xtotal=xtotal+i;ytotal=ytotal+j;count=count+1
        SourceList=[];DestList=[]
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if LL[i][j].isactive==ACTIVE:
                    x0=(xtotal+ytotal)/count;y0=(ytotal-xtotal )/count
                    xr=(xtotal+ytotal)%count;yr=(ytotal-xtotal)%count
                    x=x0-j;y=y0+i
                    if xr>=count/2:x=x+1
                    if yr>=count/2:y=y+1
                    SourceList.append([i,j]);DestList.append([x,y])
                    if x<0 or x>=HEIGHT or y<0 or y>=WIDTH:Moveable=FALSE
                    if x>=0 and x<HEIGHT and y>=0 and y<WIDTH and BL[x][y]==1 and LL[x][y].isactive==PASSIVE:Moveable=FALSE
        if Moveable==TRUE:
            for i in range(len(SourceList)):
                self.Empty(SourceList[i][0],SourceList[i][1])
            for i in range(len(DestList)):
                self.Fill(DestList[i][0],DestList[i][1])
    def Left(self,event):
        BL=self.BlockList;LL=self.LabelList
        Moveable=TRUE
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if LL[i][j].isactive==ACTIVE and j-1<0:Moveable=FALSE
                if LL[i][j].isactive==ACTIVE and j-1>=0 and BL[i][j-1]==1 and LL[i][j-1].isactive==PASSIVE:Moveable=FALSE
        if Moveable==TRUE:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if j-1>=0 and LL[i][j].isactive==ACTIVE and BL[i][j-1]==0:
                        self.Fill(i,j-1);self.Empty(i,j)
    def Right(self,event):
        BL=self.BlockList;LL=self.LabelList
        Moveable=TRUE
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if LL[i][j].isactive==ACTIVE and j+1>=WIDTH:Moveable=FALSE
                if LL[i][j].isactive==ACTIVE and j+1<WIDTH and BL[i][j+1]==1 and LL[i][j+1].isactive==PASSIVE:Moveable=FALSE
        if Moveable==TRUE:
            for i in range(HEIGHT-1,-1,-1):
                for j in range(WIDTH-1,-1,-1):
                    if j+1<WIDTH and LL[i][j].isactive==ACTIVE and BL[i][j+1]==0:
                        self.Fill(i,j+1);self.Empty(i,j)
    def Space(self,event):
        while 1:
            if self.Down(0)==FALSE:break
    def OnTimer(self):
        if self.isStart==TRUE and self.isPause==FALSE:
            self.TotalTime = self.TotalTime + float(self.time)/1000
            self.SpendTime.config(text=str(self.TotalTime))
        
        if self.isPause==FALSE:
            self.Down(0)
        if self.TotalScore>=1000:self.time=900
        if self.TotalScore>=2000:self.time=750
        if self.TotalScore>=3000:self.time=600
        if self.TotalScore>=4000:self.time=400
        self.after(self.time,self.OnTimer)
    def Down(self,event):
        BL=self.BlockList;LL=self.LabelList
        Moveable=TRUE
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if LL[i][j].isactive==ACTIVE and i+1>=HEIGHT:Moveable=FALSE
                if LL[i][j].isactive==ACTIVE and i+1<HEIGHT and BL[i+1][j]==1 and LL[i+1][j].isactive==PASSIVE:Moveable=FALSE
        if Moveable==TRUE:
            for i in range(HEIGHT-1,-1,-1):
                for j in range(WIDTH-1,-1,-1):
                    if i+1<HEIGHT and LL[i][j].isactive==ACTIVE and BL[i+1][j]==0:
                        self.Fill(i+1,j);self.Empty(i,j)
        if Moveable==FALSE:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    LL[i][j].isactive=PASSIVE
            self.JudgeLineFill()
            self.Start()
            if self.isgameover==TRUE:showinfo('T_T','The game is over!');self.Distroy();return FALSE
            for i in range(4):
                for j in range(4):
                    self.NextEmpty(i,j)
            self.Rnd()
        return Moveable
    def JudgeLineFill(self):
        BL=self.BlockList;LL=self.LabelList
        count=0;LineList=[]
        for i in range(WIDTH):LineList.append(1)
        #display flash
        for i in range(HEIGHT):
            if BL[i]==LineList:
                count=count+1
                for k in range(WIDTH):
                    LL[i][k].config(bg=str(self.flashg))
                    LL[i][k].update()
        if count!=0:self.after(100)
        #delete block
        for i in range(HEIGHT):
            if BL[i]==LineList:
                #count=count+1
                for j in range(i,0,-1):
                    for k in range(WIDTH):
                        BL[j][k]=BL[j-1][k]
                        LL[j][k]['relief']=LL[j-1][k].cget('relief')
                        LL[j][k]['bg']=LL[j-1][k].cget('bg')
                for l in range(WIDTH):
                    BL[0][l]=0
                    LL[0][l].config(relief=FLAT,bg=str(self.backg))
        self.TotalLine=self.TotalLine+count
        if count==1:self.TotalScore=self.TotalScore+1*WIDTH
        if count==2:self.TotalScore=self.TotalScore+3*WIDTH
        if count==3:self.TotalScore=self.TotalScore+6*WIDTH
        if count==4:self.TotalScore=self.TotalScore+10*WIDTH
        self.Line.config(text=str(self.TotalLine))
        self.Score.config(text=str(self.TotalScore))
    def Fill(self,i,j):
        if j<0:return
        if self.BlockList[i][j]==1:self.isgameover=TRUE
        self.BlockList[i][j]=1
        self.LabelList[i][j].isactive=ACTIVE
        self.LabelList[i][j].config(relief=RAISED,bg=str(self.frontg))
    def Empty(self,i,j):
        self.BlockList[i][j]=0
        self.LabelList[i][j].isactive=PASSIVE
        self.LabelList[i][j].config(relief=FLAT,bg=str(self.backg))
    def Play(self,event):
        showinfo('Made in China','''^_</font></p>
<p><span mce_name="em" style="font-style: italic;" class="Apple-style-span" mce_style="font-style: italic;"><span style="font-size: small; " id="" mce_style="font-size: small;"><br></span></span></p>
<p><span mce_name="em" style="font-style: italic;" class="Apple-style-span" mce_style="font-style: italic;"><span style="font-size: small; " id="" mce_style="font-size: small;">&nbsp;&nbsp; &nbsp;</span></span></p>
<p><br></p>''')
    def NextFill(self,i,j):
        self.NextList[i][j].config(relief=RAISED,bg=str(self.frontg))
    def NextEmpty(self,i,j):
        self.NextList[i][j].config(relief=FLAT,bg=str(self.nextg))
    def Distroy(self):
        #save
        if self.TotalScore!=0:
            savestr='%-9u%-8u%-8.2f%-14.2f%-13.2f%-14.2f%s/n' % (self.TotalScore,self.TotalLine,self.TotalTime
                                               ,self.TotalScore/self.TotalTime
                                               ,self.TotalLine/self.TotalTime
                                               ,float(self.TotalScore)/self.TotalLine
                                               ,time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
            self.file.seek(0,2)
            self.file.write(savestr)
            self.file.flush()
        
        for i in range(HEIGHT):
            for j in range(WIDTH):
                self.Empty(i,j)
        self.TotalLine=0;self.TotalScore=0;self.TotalTime=0.0
        self.Line.config(text=str(self.TotalLine))
        self.Score.config(text=str(self.TotalScore))
        self.SpendTime.config(text=str(self.TotalTime))
        self.isgameover=FALSE
        self.isStart=FALSE
        self.time=1000
        for i in range(4):
            for j in range(4):
                self.NextEmpty(i,j)
    def Start(self):
        if self.x==1:self.Fill(0,WIDTH/2-2);self.Fill(0,WIDTH/2-1);self.Fill(0,WIDTH/2);self.Fill(0,WIDTH/2+1)
        if self.x==2:self.Fill(0,WIDTH/2-1);self.Fill(0,WIDTH/2);self.Fill(1,WIDTH/2-1);self.Fill(1,WIDTH/2)
        if self.x==3:self.Fill(0,WIDTH/2);self.Fill(1,WIDTH/2-1);self.Fill(1,WIDTH/2);self.Fill(1,WIDTH/2+1)
        if self.x==4:self.Fill(0,WIDTH/2-1);self.Fill(1,WIDTH/2-1);self.Fill(1,WIDTH/2);self.Fill(1,WIDTH/2+1)
        if self.x==5:self.Fill(0,WIDTH/2+1);self.Fill(1,WIDTH/21);self.Fill(1,WIDTH/2);self.Fill(1,WIDTH/2+1)
        if self.x==6:self.Fill(0,WIDTH/2-1);self.Fill(0,WIDTH/2);self.Fill(1,WIDTH/2);self.Fill(1,WIDTH/2+1)
        if self.x==7:self.Fill(0,WIDTH/2);self.Fill(0,WIDTH/2+1);self.Fill(1,WIDTH/2-1);self.Fill(1,WIDTH/2)
        self.isStart=TRUE
    def Rnd(self):
        self.x=random.randint(1,7)
        if self.x==1:self.NextFill(0,0);self.NextFill(0,1);self.NextFill(0,2);self.NextFill(0,3)
        if self.x==2:self.NextFill(0,1);self.NextFill(0,2);self.NextFill(1,1);self.NextFill(1,2)
        if self.x==3:self.NextFill(0,2);self.NextFill(1,1);self.NextFill(1,2);self.NextFill(1,3)
        if self.x==4:self.NextFill(0,1);self.NextFill(1,1);self.NextFill(1,2);self.NextFill(1,3)
        if self.x==5:self.NextFill(0,3);self.NextFill(1,1);self.NextFill(1,2);self.NextFill(1,3)
        if self.x==6:self.NextFill(0,1);self.NextFill(0,2);self.NextFill(1,2);self.NextFill(1,3)
        if self.x==7:self.NextFill(0,2);self.NextFill(0,3);self.NextFill(1,1);self.NextFill(1,2)
    def RndFirst(self):
        self.x=random.randint(1,7)
    def Show(self):
        self.file.seek(0)
        strHeadLine=self.file.readline()
        dictLine={}
        strTotalLine=''
        for OneLine in self.file.readlines():
            temp=int(OneLine[:5])
            dictLine[temp]=OneLine
            
        list=sorted(dictLine.items(),key=lambda d:d[0])
        ii=0        
        for onerecord in reversed(list):
            ii=ii+1
            if ii<11:
                strTotalLine+=onerecord[1]
        showinfo('Ranking', strHeadLine+strTotalLine)
def Start():
    app.RndFirst();app.Start();app.Rnd()
def End():
    app.Distroy()
def Set():
    pass
def Show():
    app.Show()

mainmenu=Menu(root)
root['menu']=mainmenu
gamemenu=Menu(mainmenu)
mainmenu.add_cascade(label='game',menu=gamemenu)
gamemenu.add_command(label='start',command=Start)
gamemenu.add_command(label='end',command=End)
gamemenu.add_separator()
gamemenu.add_command(label='exit',command=root.quit)
setmenu=Menu(mainmenu)
mainmenu.add_cascade(label='set',menu=setmenu)
setmenu.add_command(label='set',command=Set)
showmenu=Menu(mainmenu)
mainmenu.add_cascade(label='show',menu=showmenu)
showmenu.add_command(label='show',command=Show)
app=App(root)
root.mainloop()
