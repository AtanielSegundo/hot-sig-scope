import raylib as rl

class ScreenCfg:
    default_width  = 1280
    default_height = 720
    window_title   = bytes("Scope", encoding="UTF-8")
    fresh          = True

    @classmethod
    def width(cls) -> int:
        return cls.default_width if cls.fresh else rl.GetScreenWidth()

    @classmethod
    def height(cls) -> int:
        return cls.default_height if cls.fresh else rl.GetScreenHeight()

    @classmethod
    def consume_fresh(cls) -> None:
        cls.fresh = False