
#import tkinter as tk
import pyglet as pgl
from time import sleep

from PIL import ImageTk
# from numpy import array
# from pkg_resources import resource_string as resource_bytes

# from flatland.utils.graphics_layer import GraphicsLayer
from flatland.utils.graphics_pil import PILSVG


class PGLGL(PILSVG):
    window = pgl.window.Window()
    singleton = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_open = False
        self.__class__.singleton = self

    def open_window(self):
        print("open_window - pyglet")
        assert self.window_open is False, "Window is already open!"
        #self.__class__.window.title("Flatland")
        #self.__class__.window.configure(background='grey')
        self.window_open = True

    def close_window(self):
        self.panel.destroy()
        # quit but not destroy!
        self.__class__.window.quit()

    def show(self, block=False):
        # print("show - ", self.__class__)
        pil_img = self.alpha_composite_layers()

        if not self.window_open:
            self.open_window()


        #tkimg = ImageTk.PhotoImage(img)

        # convert our PIL image to pyglet:
        bytes_image = pil_img.tobytes()
        pgl_image = pgl.image.ImageData(pil_img.width, pil_img.height, 'RGBA',
            bytes_image,
            pitch=-pil_img.width * 4)

        pgl_image.blit(0,0)

        #print("pyglet event loop 1")
        # custom event loop for Pyglet
        pgl.clock.tick()
        #print("pyglet event loop 2")
        for window in pgl.app.windows:
            #print("pyglet event loop3")
            window.switch_to()
            window.dispatch_events()
            #window.dispatch_event('on_draw')
            #print("pyglet event loop 4 ")
            window.flip()
            #print("pyglet event loop 5")


        #self.__class__.window.update()
        self.firstFrame = False

@PGLGL.window.event
def on_draw():
    print("pyglet event loop 6")
    PGLGL.window.clear()
    self = PGLGL.singleton
    #self.show()
    print("pyglet event loop7")
    

def test_pyglet():
    oGL = PGLGL(400,300)
    sleep(2)


if __name__=="__main__":
    test_pyglet()