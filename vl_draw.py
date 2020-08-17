from threading import Thread, Event
from visualdl import *
import shutil

class DrawScalar(Thread):

    def __init__(self, log_path, tag="test"):
        super(DrawScalar, self).__init__()
        self.event = Event()
        self.log_path = log_path
        self.tag = tag
        self.stop_flag = False
        self.epoch = None
        self.value = None
        try:
            shutil.rmtree(self.log_path)
        except:
            pass
    
    def set_value(self, epoch, value):
        self.epoch = epoch
        self.value = value
        self.event.set()

    def stop(self):
        self.stop_flag = True
        self.event.set()

    def run(self):
        with LogWriter(logdir=self.log_path) as writer:
            while not self.stop_flag:
                self.event.wait()
                self.event.clear()
                if not self.stop_flag:
                    writer.add_scalar(tag=self.tag+"/acc", step=self.epoch, value=self.value['acc'])
                    writer.add_scalar(tag=self.tag+"/loss", step=self.epoch, value=self.value['loss'])
        print("DrawScalar exited...")