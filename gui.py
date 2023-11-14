from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.video import Video
from kivy.uix.camera import Camera
from kivy.uix.videoplayer import VideoPlayer

class MyApp(App):
    def build(self):
        # Diseño principal
        layout_principal = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Sub-layouts para las dos secciones superiores
        layout_superior = BoxLayout(orientation='horizontal', spacing=10)
        layout_izquierdo = BoxLayout(orientation='vertical', spacing=10)
        layout_derecho = BoxLayout(orientation='vertical', spacing=10)

        # Placeholder para la imagen de la webcam
        cam_webcam = Camera(play=True)
        layout_izquierdo.add_widget(cam_webcam)

        # Video en el cuadro derecho
        video_source = 'video1.mp4'  # Reemplaza con la ruta de tu video
        video = Video(source=video_source, state='play', options={'eos': 'loop'}, size_hint=(1, 1))
        layout_derecho.add_widget(video)

        # Botones en la parte inferior con orientación horizontal
        layout_botones = BoxLayout(orientation='horizontal', spacing=10)
        btn_funcion_1 = Button(text='Función 1', on_press=self.placeholder_funcion_1)
        btn_funcion_2 = Button(text='Función 2', on_press=self.placeholder_funcion_2)
        btn_funcion_3 = Button(text='Función 3', on_press=self.placeholder_funcion_3)

        # Agregar widgets a los layouts
        layout_superior.add_widget(layout_izquierdo)
        layout_superior.add_widget(layout_derecho)

        layout_botones.add_widget(btn_funcion_1)
        layout_botones.add_widget(btn_funcion_2)
        layout_botones.add_widget(btn_funcion_3)

        layout_principal.add_widget(layout_superior)
        layout_principal.add_widget(layout_botones)

        return layout_principal

    def placeholder_funcion_1(self, instance):
        print("Función 1 activada")

    def placeholder_funcion_2(self, instance):
        print("Función 2 activada")

    def placeholder_funcion_3(self, instance):
        print("Función 3 activada")

if __name__ == '__main__':
    MyApp().run()
