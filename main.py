from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.gridlayout import GridLayout

from kivy.properties import ObjectProperty
from kivy.utils import get_color_from_hex
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.clock import Clock
from sarcasm import Sarcasm

#background color
Window.clearcolor = get_color_from_hex('#16203B')

#register font
LabelBase.register(name='Roboto',fn_regular="./fonts/Roboto-Thin.ttf",fn_bold='./fonts/Roboto-Medium.ttf')

class DetectSarcasmRoot(BoxLayout):
    '''
    Root of all widgets
    '''
    start_screen = ObjectProperty(None)

    def __init__(self,**kwargs):
        super(DetectSarcasmRoot,self).__init__(**kwargs)
        self.sarcasm = Sarcasm()
        

    # call baseline 1
    def baseline1(self):
        return self.sarcasm.baseline1()

    # call baseline 2 with n_gram
    def baseline2(self,gram):
        return self.sarcasm.baseline2(gram)
    # call baseline 3 of sarcassm
    def baseline3(self):
        return self.sarcasm.baseline3()

    def process(self,baseline,gram):
        '''
        Process button 
        '''
        if baseline == 'Contradiction':
            confusion_matix,classification_report,accuracy = self.baseline1()
        elif baseline == 'Contra + coherence':
            confusion_matix,classification_report,accuracy = self.baseline3()
        elif baseline == 'N_gram':
            print(gram[0])
            print('Started N_gram Processing')
            self.start_screen.process.text = 'WAIT'
        
            confusion_matix,classification_report,accuracy = self.baseline2(int(gram[0]))
            self.start_screen.process.text = 'Process'
        
        #set accuracy button
        self.start_screen.accuracy.text = str(accuracy)
        #set confusion matix
        self.start_screen.tp.text = str(confusion_matix[0][0])
        self.start_screen.fn.text = str(confusion_matix[0][1])
        self.start_screen.fp.text = str(confusion_matix[1][0])
        self.start_screen.tn.text = str(confusion_matix[1][1])
    
        # set classification matrix
        self.start_screen.classification.text = classification_report


    
# class ConfusionMatrix(GridLayout):
#     '''
#     ConfusionMatrix Layout
#     '''
#     def __init__(self,*args,**kwargs):
#         super(ConfusionMatrix,self).__init__(*args,**kwargs)
#         self.cols = 2
#         self.spacing: 10
#         self.createButtons()
#         self.size_hint_x: 15

    
    # def createButtons(self):
    #     _list = ["TP",'FN',"FP","TN"]
    #     for n in _list:
    #         self.add_widget(Button(text=str(n),id=n))



class DetectSarcasmApp(App):
    '''
    '''

    def __init__(self,**kwargs):
        super(DetectSarcasmApp,self).__init__(**kwargs)

    def build(self):
        return DetectSarcasmRoot()


if __name__ == '__main__':
    DetectSarcasmApp().run()
