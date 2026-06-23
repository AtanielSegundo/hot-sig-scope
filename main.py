import time
import traceback
import raylib as rl
from core.hot_reload import HotReloader

import core.render   as render
import core.configs  as configs
import core.scope    as scope
import core.draw     as draw
import core.fir_core as fir_core

def when_configs_reloaded() -> None:
    rl.SetWindowSize(configs.ScreenCfg.width(), configs.ScreenCfg.height())
    rl.SetWindowTitle(configs.ScreenCfg.window_title)
    configs.ScreenCfg.consume_fresh()

def main() -> None:
    hot_reloader = HotReloader(
        modules_root = "./core",
        targets = {
            "render.py"   : render,
            "scope.py"    : scope,
            "draw.py"     : draw,
            "configs.py"  : configs,
            "fir_core.py" : fir_core,
        }
    )    
    
    rl.SetConfigFlags(rl.FLAG_WINDOW_RESIZABLE)
    rl.InitWindow(configs.ScreenCfg.width(),
                  configs.ScreenCfg.height(),
                  configs.ScreenCfg.window_title)
    when_configs_reloaded()
    
    rl.SetTargetFPS(60)
    
    state = dict()

    while not rl.WindowShouldClose():
        rl.BeginDrawing()
        try:
            if hot_reloader.is_module_updated(configs):
                when_configs_reloaded()
            
            with hot_reloader:
                #configs.render_target(configs,state)
                render.render_target(configs,state)
                
        except Exception as e:
            print(e)
            traceback.print_exc()
            time.sleep(2)

        rl.EndDrawing()


if __name__ == "__main__":
    main()