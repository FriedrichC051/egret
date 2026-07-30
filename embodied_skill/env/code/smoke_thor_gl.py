"""AI2-THOR OpenGL(Linux64) 冒烟测试：走 WSLg 的 GPU(d3d12) OpenGL，而非 Vulkan/CloudRendering。

运行： wsl -d Ubuntu -- bash embodied_skill/env/scripts/wsl_smoke_gl.sh
"""
import time
from ai2thor.controller import Controller
from ai2thor.platform import Linux64


def main():
    t0 = time.time()
    controller = Controller(
        platform=Linux64,
        scene="FloorPlan1",
        width=300,
        height=300,
        x_display="0",  # 用 WSLg 的 DISPLAY=:0
    )
    print("INIT_OK elapsed=%.1fs" % (time.time() - t0))
    event = controller.step("MoveAhead")
    print("SMOKE_OK", event.metadata["lastActionSuccess"])
    # 打印一帧尺寸，确认真的渲染了画面
    if event.frame is not None:
        print("FRAME", event.frame.shape)
    controller.stop()


if __name__ == "__main__":
    main()
