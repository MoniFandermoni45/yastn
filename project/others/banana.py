import time
from pycallgraph2 import PyCallGraph
from pycallgraph2 import Config, GlobbingFilter
from pycallgraph2.output import GraphvizOutput

class Banana:

    def __init__(self):
        pass
    
    def eat(self):
        self.secret_function()
        self.chew()
        self.swallow()
    
    def secret_function(self):
        time.sleep(0.2)
    
    def chew(self):
        pass

    def swallow(self):
        pass

config = Config(max_depth=1)
config.trace_filter = GlobbingFilter(
    exclude=[
        'pycallgraph.*',
        '*.secret_function'
    ]
)

graphviz = GraphvizOutput(output_file='filter_max_depth.png')
with PyCallGraph(output=graphviz, config=config):
    banana = Banana()
    banana.eat()