import mujoco

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 .1"/>
    <body pos="0 0 .2"><freejoint/><geom type="box" size=".05 .05 .05" rgba="1 0 0 1"/></body>
  </worldbody>
</mujoco>
"""


def test_offscreen_render():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    with mujoco.Renderer(model, 240, 320) as renderer:
        renderer.update_scene(data)
        img = renderer.render()
    assert img.shape == (240, 320, 3)
    assert img.mean() > 0
