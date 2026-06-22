import os
import time
import traceback
import threading

from typing  import *
from pathlib import Path

from watchdog.observers import Observer
from importlib       import reload
from watchdog.events import FileSystemEventHandler

class HotReloadHandler(FileSystemEventHandler):
    def __init__(self,sych_semaphore,fn_on_modified:callable):
        self.fn_on_modified = fn_on_modified
        self.sych_semaphore = sych_semaphore
        super().__init__()
    def on_modified(self, event):
        self.fn_on_modified(event)

class HotReloader:    
    def __init__(self,
                 modules_root:str,
                 targets:Dict[List[Path],Callable[[Any],Any]],
                 reload_debounce_s = 0.3                 
                 ):
        self.modules_root = modules_root
        if not Path(self.modules_root).exists():
            raise FileNotFoundError(f"{self.modules_root}")

        self.targets     = targets
        self.debounce_s  = reload_debounce_s
        self.module_updated = {
            self.get_module_last_name(val): False 
            for val in targets.values()
        }
        self.last_reload = dict()
        self.guard       = threading.Lock()
        self.observer    = Observer()
        self.handler     = HotReloadHandler(self.guard,
                                        self.on_modified_callback)
        
        self.observer.schedule(self.handler,self.modules_root)
        self.observer.start()

    @staticmethod
    def get_module_last_name(module) -> str:
        return module.__name__.split(".")[-1]
        
    def on_modified_callback(self,event):
        if event.is_directory:
            return
        
        with self.guard:
            name   = os.path.basename(event.src_path)
            target = self.targets.get(name)
            if target is None:
                return
            
            self.module_updated[self.get_module_last_name(target)] = True    
            
            now = time.monotonic()
            if now - self.last_reload.get(name, 0.0) < self.debounce_s:
                return                  
            self.last_reload[name] = now

            try:
                reload(target)
            except Exception as e:
                print("[HOT-RELOAD] Failed, Keeping Old Version")
                print(f"{e}\n")
                traceback.print_exc()

    def is_module_updated(self,module):
        module_name  = self.get_module_last_name(module)
        updated_flag = self.module_updated.get(module_name,None)
        if updated_flag is None:
            return False
        
        elif updated_flag:
            self.module_updated[module_name] = False
            
        return updated_flag
    
    def __enter__(self):
        self.guard.__enter__()
        return self

    def __exit__(self, *args):
        self.guard.__exit__(*args)