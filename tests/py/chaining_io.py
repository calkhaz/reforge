color_adjust = dict(
    saturation = 1.5,
    contrast = 1.2,
    brightness = -0.2,
    hue_adjust = 90.0 # 2PI brings you back to 0.0
)

nodes = dict(
    color_adjust = color_adjust,
)

graph = '''input -> color_adjust -> combination:input_image0
           input -> passthrough  -> combination:input_image1
                    combination  -> output'''
