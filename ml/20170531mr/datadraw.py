# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import pylab

import dataset
import datadraw

class DataDraw():
    ax=''
    colors = {
        'aliceblue':            '#F0F8FF',
        'antiquewhite':         '#FAEBD7',
        'aqua':                 '#00FFFF',
        'aquamarine':           '#7FFFD4',
        'azure':                '#F0FFFF',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'black':                '#000000',
        'blanchedalmond':       '#FFEBCD',
        'blue':                 '#0000FF',
        'blueviolet':           '#8A2BE2',
        'brown':                '#A52A2A',
        'burlywood':            '#DEB887',
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',
        'coral':                '#FF7F50',
        'cornflowerblue':       '#6495ED',
        'cornsilk':             '#FFF8DC',
        'crimson':              '#DC143C',
        'cyan':                 '#00FFFF',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'floralwhite':          '#FFFAF0',
        'forestgreen':          '#228B22',
        'fuchsia':              '#FF00FF',
        'gainsboro':            '#DCDCDC',
        'ghostwhite':           '#F8F8FF',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'indigo':               '#4B0082',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'lime':                 '#00FF00',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navajowhite':          '#FFDEAD',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olive':                '#808000',
        'olivedrab':            '#6B8E23',
        'orange':               '#FFA500',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'silver':               '#C0C0C0',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'teal':                 '#008080',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'wheat':                '#F5DEB3',
        'white':                '#FFFFFF',
        'whitesmoke':           '#F5F5F5',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'
    }

    def __init__(self,typex='3d'):
        if typex=="3d":
            fig = plt.figure()  
            self.ax = fig.add_subplot(111, projection='3d') 

    def getColorsValue(self):
        colors=self.colors
        keys=colors.keys()
        colors_tmp=[colors[key] for key in keys]
        return colors_tmp

    def drawline(self,data,c='r'):
        x=data[0]
        y=data[1]
        plt.plot(x,y,c=c)

    def drawgoal(self,data,c='r'):
        plt.scatter(data[0],data[1],c=c)

    def drawbatchgoal(self,data,c='r'):
        plt.scatter(data[:,0],data[:,1],c=c)

    def draw(self,data,fname='./data/a.png',save=False):
        x=data[0]
        y=data[1]
        plt.plot(x,y)
        if save:
            plt.savefig(fname)
            plt.clf()
            plt.close()
        else:
            plt.show()

    def draw3dline(self,data,ax='',c='r'):
        if ax=='':
            ax=self.ax
        X=data[0]
        Y=data[1]
        Z=data[2]
        ax.plot(X,Y,Z,c=c)

    def draw3dgoal(self,data,ax='',c='r'):
        if ax=='':
            ax=self.ax
        ax.scatter(data[0],data[1],data[2],c=c)

def draw3d():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw()

    START=2700
    PAIRS=2
    colors=['b','r','g','y','c','k','m']
    for i in range(PAIRS):
        dw.draw3dline(mouses[i])
        dw.draw3dline(mouses[START+i])
        dw.draw3dgoal([goals[i][0],goals[i][1],i],c=colors[i%7])
        dw.draw3dgoal([goals[START+i][0],goals[START+i][1],START+i],c=colors[(i+3)%7])
    plt.show()

def draw2d():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw("2d")

    START=2700
    PAIRS=2
    colors=['b','r','g','y','c','k','m']
    for i in range(PAIRS):
        dw.drawline(mouses[i])
        dw.drawline(mouses[START+i])
        # dw.drawgoal([goals[i][0],goals[i][1],i],c=colors[i%7])
        # dw.drawgoal([goals[START+i][0],goals[START+i][1]],c=colors[(i+3)%7])
    plt.show()

def drawScatter():
    ds=dataset.DataSet()
    ds.getTrainData()
    mouses=ds.train["mouses"]
    goals=ds.train["goals"]
    dw=datadraw.DataDraw("2d")
    mouses_start=ds.getPosOfMouse(1)
    dw.drawbatchgoal(mouses_start[:2600],'y')
    dw.drawbatchgoal(mouses_start[2600:],'b')

    dw.drawbatchgoal(goals[:2600],'y')
    dw.drawbatchgoal(goals[2600:],'b')
    
    plt.show()

def plot_confusion_matrix(cm, genre_list, name, title,max,save=False):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Greens', vmin=0, vmax=max)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    cm=cm.T
    for i in range(len(cm)):
        # t=len(cm[0])-i
        for j in range(len(cm[0])):
            if cm[i,j]<1e-2:
                continue
            pylab.text(i, j, '%.2f'%cm[i,j])

    pylab.title(title)
    pylab.colorbar()
    pylab.grid(True)
    pylab.show()
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(True)
    if save==True:
        pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png"%name), bbox_inches="tight")


if __name__=="__main__":
   draw3d()
   # draw2d()
   # drawScatter()