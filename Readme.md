

<DetectSarcasmRoot>:
    orientation: 'vertical'
    kivy_screen_manager: kivy_screen_manager
    padding: 10
    spacing: 10
    BoxLayout:

        ActionBar:
            pos_hint: {'top':1}
            ActionView:
                ActionPrevious:
                    title: "Sarcasm Detection"
                    with_previous: False
                ActionOverflow:
                    ActionButton:
                        text: "Settings"
                        on_press: app.open_settings()
    
        ScreenManager:
            id: kivy_screen_manager
            StartScreen:
                name: "start_screen"

<StartScreen@Screen>:
    BoxLayout:
        size_hint_y: None
        height: "40dp"
        Spinner:
            id: baseline
            text: "Contradiction"
            values: ('Contradiction','N_gram')
            # size_hint: None, None
            size_hint_x: 15
        Spinner:
            id: n_gram
            text: '1 gram'
            values: ('1 gram','2 gram', '3 gram')
            size_hint_x: 15
        Button:
            text:'Process'
            on_press: app.root.process(baseline.text,n_gram.text)
            size_hint_x:15


    GridLayout:
        rows: 4
        cols: 3
        size_hint_y: None
        height: "180dp"
        border: (2,2,2,2,2)
        Label:
            text: "Confusion Matrix"
        Button:
            text:"Yes"




            
            



